import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
from templateflow import api as templateflow
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import PMC, MTL, subject_id, export

ROI_SPEC = [
    ("Precuneus", "cortical", ["Precuneous Cortex", "Precuneus"]),
    ("PCC", "cortical", ["Cingulate Gyrus, posterior division", "Posterior Cingulate"]),
    ("Anterior PHG", "cortical", ["Parahippocampal Gyrus, anterior division"]),
    ("Posterior PHG", "cortical", ["Parahippocampal Gyrus, posterior division"]),
    ("Hippocampus", "subcortical", ["Hippocampus"]),
    ("Amygdala", "subcortical", ["Amygdala"]),
]
MOTION = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
SPACE = "MNI152NLin2009cAsym"


def roi_atlas(out):
    cort_path = templateflow.get(SPACE, resolution=2, atlas="HOCPA", desc="th25", suffix="dseg", extension=".nii.gz")
    sub_path = templateflow.get(SPACE, resolution=2, atlas="HOSPA", desc="th25", suffix="dseg", extension=".nii.gz")
    cort = nib.load(str(cort_path))
    sub = nib.load(str(sub_path))
    if cort.shape != sub.shape or not np.allclose(cort.affine, sub.affine):
        sub = image.resample_to_img(sub, cort, interpolation="nearest", force_resample=True, copy_header=True)
    images = {"cortical": np.asarray(cort.dataobj), "subcortical": np.asarray(sub.dataobj)}
    labels = {
        "cortical": datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm").labels,
        "subcortical": datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm").labels,
    }
    atlas = np.zeros(cort.shape, dtype=np.int16)
    definitions = []
    for index, (name, region, patterns) in enumerate(ROI_SPEC, start=1):
        indices = [i for i, label in enumerate(labels[region])
                   if any(pattern.lower() in str(label).lower() for pattern in patterns)]
        mask = np.isin(images[region], indices)
        atlas[mask] = index
        definitions.append({"region": name, "atlas": region, "indices": ",".join(map(str, indices)),
                            "voxels": int(mask.sum())})
    out.mkdir(parents=True, exist_ok=True)
    atlas_path = out / "harvard_oxford_pmc_mtl.nii.gz"
    nib.save(nib.Nifti1Image(atlas, cort.affine, cort.header), atlas_path)
    export(pd.DataFrame(definitions), out / "roi_definitions.csv")
    return atlas_path


def confounds_table(path):
    raw = pd.read_csv(path, sep="\t")
    columns = [col for col in MOTION if col in raw]
    for base in [*MOTION, "white_matter", "csf", "global_signal"]:
        if base in ["white_matter", "csf", "global_signal"] and base in raw:
            columns.append(base)
        for suffix in ["derivative1", "power2", "derivative1_power2"]:
            col = f"{base}_{suffix}"
            if col in raw:
                columns.append(col)
    columns += [col for col in raw if col.startswith("non_steady_state_outlier")]
    nuisance = raw[columns].replace([np.inf, -np.inf], np.nan).fillna(0).copy()
    t = np.linspace(-1, 1, len(raw))
    nuisance["linear_trend"] = t
    nuisance["quadratic_trend"] = t**2
    scales = nuisance.std(ddof=1).replace(0, 1).fillna(1)
    nuisance = (nuisance - nuisance.mean()) / scales
    fd = raw["framewise_displacement"].fillna(0).to_numpy(dtype=float)
    valid = np.ones(len(raw), dtype=bool)
    valid[:10] = False
    for index in np.flatnonzero(fd >= 1.0):
        valid[max(0, index - 2):min(len(raw), index + 3)] = False
    return nuisance, np.flatnonzero(valid), float(np.mean(fd))


def matched_confounds(bold):
    stem = bold.name.split("_space-")[0]
    candidates = sorted(bold.parent.glob("*desc-confounds_timeseries.tsv"))
    return next(p for p in candidates if p.name.startswith(stem))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmriprep", required=True)
    parser.add_argument("--out", default="./results")
    parser.add_argument("--task", default="thinkaloud")
    parser.add_argument("--exclude-file", default="config/imaging_exclusions.txt")
    parser.add_argument("--subjects-file", default=None)
    args = parser.parse_args()

    root = Path(args.fmriprep)
    for candidate in (root, root / "fmriprep", root / "derivatives", root / "derivatives" / "fmriprep"):
        if candidate.is_dir() and any(candidate.glob("sub-*/func")):
            root = candidate
            break
    out = Path(args.out)
    atlas = roi_atlas(out / "atlas")
    excluded = {subject_id(x.strip()) for x in Path(args.exclude_file).read_text().splitlines() if x.strip()}
    included = None
    if args.subjects_file:
        included = {subject_id(x.strip()) for x in Path(args.subjects_file).read_text().splitlines() if x.strip()}
    mask = NiftiLabelsMasker(labels_img=str(atlas), labels=["Background", *[x[0] for x in ROI_SPEC]],
                             background_label=0, standardize="zscore_sample", standardize_confounds=False,
                             detrend=True, smoothing_fwhm=4.0, t_r=1.5, resampling_target="data")
    bold_files = sorted(root.glob(f"sub-*/func/*task-{args.task}*space-{SPACE}*desc-preproc_bold.nii.gz"))
    rows = []
    qc = []
    if not bold_files:
        raise FileNotFoundError(f"No preprocessed think-aloud BOLD files found in {root}")
    for bold in tqdm(bold_files, desc="PMC-MTL connectivity"):
        subject = subject_id(bold.parent.parent.name)
        if subject in excluded or (included is not None and subject not in included):
            continue
        nuisance, sample_mask, mean_fd = confounds_table(matched_confounds(bold))
        timeseries = mask.fit_transform(str(bold), confounds=nuisance, sample_mask=sample_mask)
        corr = np.clip(np.corrcoef(timeseries, rowvar=False), -0.999999, 0.999999)
        z = np.arctanh(corr)
        for i, pmc in enumerate(PMC):
            for j, mtl in enumerate(MTL, start=2):
                rows.append({"subject": subject, "run": bold.name, "edge": f"{pmc}--{mtl}",
                             "signed_fisher_z_fc": float(z[i, j])})
        qc.append({"subject": subject, "run": bold.name, "mean_fd": mean_fd,
                   "n_retained": len(sample_mask)})
    frame = pd.DataFrame(rows)
    subject_edges = frame.groupby(["subject", "edge"], as_index=False)["signed_fisher_z_fc"].mean()
    export(subject_edges, out / "tables" / "connectivity_edges.csv")
    export(pd.DataFrame(qc), out / "tables" / "connectivity_qc.csv")
    print(f"Saved signed FC for {subject_edges['subject'].nunique()} participants")


if __name__ == "__main__":
    main()
