import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import (ATQ, COVARIATES, EDGES, MTL, NODES, PMC, TARGET,
                           association, bh_adjust, bootstrap_pearson, export, partial_pearson,
                           permutation_partial_r, signed_svd, subject_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--participants", required=True)
    parser.add_argument("--out", default="./results")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    tables = out / "tables"
    participants = pd.read_csv(args.participants)
    participants["subject"] = participants["subject"].map(subject_id)
    semantic = pd.read_csv(tables / "semantic_features.csv")
    edges = pd.read_csv(tables / "connectivity_edges.csv")
    edges["subject"] = edges["subject"].map(subject_id)
    wide = edges.pivot(index="subject", columns="edge", values="signed_fisher_z_fc")
    wide = wide[EDGES].reset_index()
    data = participants.merge(semantic, on="subject", how="inner").merge(wide, on="subject", how="inner")
    data = data.dropna(subset=[ATQ, TARGET, *COVARIATES, *EDGES]).sort_values("subject").reset_index(drop=True)

    components = []
    rows = []
    for record in data.to_dict("records"):
        subject = record["subject"]
        fields, W, u, sigma, vt = signed_svd(record)
        components.append({"subject": subject, **fields})
        for mode in (1, 2):
            j = mode - 1
            for i, name in enumerate(PMC):
                rows.append({"subject": subject, "mode": mode, "partition": "PMC", "region": name,
                             "loading": float(u[i, j]), "contribution": float(u[i, j] ** 2)})
            for i, name in enumerate(MTL):
                rows.append({"subject": subject, "mode": mode, "partition": "MTL", "region": name,
                             "loading": float(vt[j, i]), "contribution": float(vt[j, i] ** 2)})
    data = data.merge(pd.DataFrame(components), on="subject", validate="one_to_one")
    export(data, tables / "analysis_dataset.csv")
    export(pd.DataFrame(rows), tables / "svd_loadings.csv")

    behavioral = pd.DataFrame([association(data, TARGET, ATQ, iterations=args.bootstrap, seed=args.seed)])
    export(behavioral, tables / "behavioral_association.csv")

    zero = []
    for feature in (TARGET, "n_sentences", "negative_sentence_prop"):
        r, p = stats.pearsonr(data[feature], data[ATQ])
        lower, upper = bootstrap_pearson(data, feature, ATQ, args.bootstrap, args.seed)
        zero.append({"feature": feature, "target": ATQ, "r": float(r), "p_raw": float(p),
                     "ci95_low": lower, "ci95_high": upper, "n": len(data)})
    zero = pd.DataFrame(zero)
    zero["p_fdr"] = bh_adjust(zero["p_raw"])
    export(zero, tables / "behavioral_zero_order.csv")

    with np.load(tables / "permutation_mnsrl.npz") as null:
        indices = {subject: i for i, subject in enumerate(null["subjects"].tolist())}
        subjects = data["subject"].tolist()
        order = [indices[subject] for subject in subjects]
        perm_rows = []
        for name in ("sentence_order", "negative_label"):
            p, distribution = permutation_partial_r(behavioral.iloc[0]["partial_r"], null[name][order],
                                                     subjects, data, ATQ, COVARIATES)
            perm_rows.append({"null_model": name, "observed_partial_r": behavioral.iloc[0]["partial_r"],
                              "empirical_p_two_sided": p, "n_permutations": len(distribution),
                              "null_mean": float(np.mean(distribution)),
                              "null_sd": float(np.std(distribution, ddof=1))})
            np.save(tables / f"{name}_null_partial_r.npy", distribution)
    export(pd.DataFrame(perm_rows), tables / "behavioral_permutation_tests.csv")

    edge = pd.DataFrame([dict(association(data, feature, iterations=args.bootstrap, seed=args.seed), edge=feature)
                         for feature in EDGES])
    edge["p_fdr"] = bh_adjust(edge["p_raw"])
    export(edge, tables / "signed_fc_edge_associations.csv")

    node_specs = [(mode, region, f"node_contrib__mode{mode}__{region.replace(' ', '_')}")
                  for mode in (1, 2) for region in NODES]
    node = pd.DataFrame([dict(association(data, feature, iterations=args.bootstrap, seed=args.seed),
                              mode=mode, region=region)
                         for mode, region, feature in node_specs])
    node["p_fdr"] = bh_adjust(node["p_raw"])
    export(node, tables / "svd_region_associations.csv")
    summary = pd.DataFrame(rows).groupby(["mode", "partition", "region"], as_index=False).agg(
        mean_contribution=("contribution", "mean"), sd_contribution=("contribution", "std"))
    export(summary, tables / "svd_region_profiles.csv")
    energy = data[["svd_decomp__mode1_matrix_energy_share", "svd_decomp__mode2_matrix_energy_share"]]
    export(pd.DataFrame({"mode": [1, 2], "mean": energy.mean().to_numpy(),
                         "sd": energy.std(ddof=1).to_numpy()}), tables / "svd_mode_energy.csv")
    pphg = node.loc[node["feature"].eq("node_contrib__mode1__Posterior_PHG")].iloc[0]
    print(f"N = {len(data)}")
    print(f"mNSRL / ATQ: r={behavioral.iloc[0]['partial_r']:.3f}")
    print(f"pPHG Mode-1 / mNSRL: r={pphg['partial_r']:.3f}; q={pphg['p_fdr']:.4f}")


if __name__ == "__main__":
    main()
