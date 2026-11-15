import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from utils import (
    create_main_figure, EDGE_ORDER, EDGE_SHORT, draw_assoc_bars,
    add_panel_label, create_figure3, format_decimal, format_p,
)


def save(fig, path, dpi):
    fig.savefig(path.with_suffix('.tif'), dpi=dpi, bbox_inches='tight', pad_inches=.05)
    fig.savefig(path.with_suffix('.png'), dpi=dpi, bbox_inches='tight', pad_inches=.05)
    plt.close(fig)


def figure1(tables, figures, dpi, bootstrap_n):
    data = pd.read_csv(tables / 'analysis_dataset.csv').rename(columns={
        'negative_Lmax_mean_10_15_20': 'observed_Lmax_mean_rrs', 'sex_M': 'sex_code'
    })
    perm = pd.read_csv(tables / 'behavioral_permutation_tests.csv').set_index('null_model')
    observed_r = float(pd.read_csv(tables / 'behavioral_association.csv').iloc[0]['partial_r'])

    def get_null(name, filename):
        arr = np.load(tables / filename)
        p = float(perm.loc[name, 'empirical_p_two_sided'])
        dist = pd.DataFrame({'r': arr})
        summary = pd.DataFrame([{
            'target': 'AutomaticThought',
            'observed_partial_r': observed_r,
            'empirical_p_two_sided_abs': p,
            'null_r_2p5': np.percentile(arr, 2.5),
            'null_r_97p5': np.percentile(arr, 97.5),
        }])
        return dist, summary

    sent_dist, sent_summary = get_null('sentence_order', 'sentence_order_null_partial_r.npy')
    label_dist, label_summary = get_null('negative_label', 'negative_label_null_partial_r.npy')
    values = create_main_figure(
        observed_subject_df=data,
        shuffled_null_dist=sent_dist,
        shuffled_null_summary=sent_summary,
        negative_order_null_dist=label_dist,
        negative_order_null_summary=label_summary,
        target='AutomaticThought', feature='observed_Lmax_mean_rrs',
        covariates=['n_sentences', 'negative_sentence_prop', 'age', 'sex_code'],
        out_dir=figures, dpi=dpi, bootstrap_n=bootstrap_n, bootstrap_seed=20260822,
    )
    values['pearson_df'].to_csv(figures / 'Figure1B_pearson.csv', index=False)
    pd.DataFrame([values['residual_stats']]).to_csv(figures / 'Figure1C_partial_residual.csv', index=False)
    values['residual_df'].to_csv(figures / 'Figure1C_residuals.csv', index=False)
    return values


def figure2(tables, figures, dpi):
    edge = pd.read_csv(tables / 'signed_fc_edge_associations.csv')
    ordered = edge.set_index('edge').reindex(EDGE_ORDER).reset_index()
    fig = plt.figure(figsize=(7.4, 5.9))
    ax = fig.add_axes([.14, .20, .82, .69])
    draw_assoc_bars(ax, ordered, [EDGE_SHORT[e] for e in EDGE_ORDER],
        'PMC-MTL edge associations with mNSRL', 'Partial '+r'$\it{r}$'+' with mNSRL',
        rotate=48, raw_sig_marker='†', use_fdr_markers=False)
    add_panel_label(ax, 'B', -.10, 1.13)
    save(fig, figures / 'Fig2', dpi)
    ordered.to_csv(figures / 'Figure2B_signed_edges.csv', index=False)


def figure3(tables, figures, dpi, bootstrap_n):
    data = pd.read_csv(tables / 'analysis_dataset.csv')
    loadings = pd.read_csv(tables / 'svd_loadings.csv').rename(columns={'region': 'node'})
    screen = pd.read_csv(tables / 'svd_region_associations.csv').rename(columns={'region': 'node'})
    create_figure3(data, loadings, screen, bootstrap_n, 42, figures, figures, dpi, False)


def figure4(tables, figures, dpi):
    med_path = tables / 'indirect_association.csv'
    if not med_path.exists():
        print('Fig4 skipped: indirect_association.csv is missing')
        return
    m = pd.read_csv(med_path).iloc[0]
    fmt = format_decimal
    fmt_p = format_p
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    navy = '#062b3d'

    def rect(cx, cy, w, h, text):
        box = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
            boxstyle='round,pad=0.012,rounding_size=.02', linewidth=2,
            edgecolor=navy, facecolor='#f3f3f3', transform=ax.transAxes)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha='center', va='center', transform=ax.transAxes, fontsize=12)

    rect(.17, .30, .28, .15, 'pPHG contribution\nto Mode-1')
    rect(.50, .77, .26, .15, 'mNSRL')
    rect(.83, .30, .25, .15, 'ATQ-30')

    def arrow(start, end, linestyle='-', lw=2):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=18,
            lw=lw, color='black', linestyle=linestyle, transform=ax.transAxes))

    arrow((.25, .39), (.44, .69))
    arrow((.57, .69), (.75, .39))
    arrow((.31, .30), (.70, .30), linestyle=':', lw=2)
    ax.text(.19, .57, f'Path a (β) = {fmt(m["path_a"])}\n{fmt_p(m["path_a_p"])}', ha='center', va='center', transform=ax.transAxes)
    ax.text(.81, .57, f'Path b (β) = {fmt(m["path_b"])}\n{fmt_p(m["path_b_p"])}', ha='center', va='center', transform=ax.transAxes)
    ax.text(.50, .36, f'Direct effect c′ (β) = {fmt(m["direct_path_c_prime"])}\n{fmt_p(m["direct_path_c_prime_p"])}', ha='center', va='bottom', transform=ax.transAxes)
    ax.text(.50, .08, f'Standardized indirect association (ACME) = {fmt(m["acme"])}\n95% CI [{fmt(m["acme_ci95_low"])}, {fmt(m["acme_ci95_high"])}], {fmt_p(m["acme_p"])}', ha='center', va='center', transform=ax.transAxes, fontsize=12)
    save(fig, figures / 'Fig4', dpi)


def write_ci_summary(tables, figures, fig1_data):
    rows = []
    fmt = format_decimal
    pear = fig1_data['pearson_df']
    for _, r in pear.iterrows():
        rows.append(['Fig1B', r['label'], fmt(r['pearson_r']), fmt(r['bootstrap_ci_low']), fmt(r['bootstrap_ci_high'])])
    rr=fig1_data['residual_stats']
    rows.append(['Fig1C', 'mNSRL vs ATQ-30, partial r', fmt(rr['partial_r']), fmt(rr['bootstrap_ci_low']), fmt(rr['bootstrap_ci_high'])])
    for _, r in pd.read_csv(tables / 'signed_fc_edge_associations.csv').iterrows():
        rows.append(['Fig2B', r['edge'], fmt(r['partial_r']), fmt(r['ci95_low']), fmt(r['ci95_high'])])
    profile_path=figures / 'Figure3BC_participant_specific_loading_summary.csv'
    if profile_path.exists():
        for _, r in pd.read_csv(profile_path).iterrows():
            rows.append([f'Fig3{"B" if int(r["mode"])==1 else "C"}', r['node'] + ' mean squared loading', fmt(r['mean_squared_loading']), fmt(r['ci95_low']), fmt(r['ci95_high'])])
    for _, r in pd.read_csv(tables / 'svd_region_associations.csv').iterrows():
        rows.append(['Fig3D/E', f'Mode-{int(r["mode"])} {r["region"]}', fmt(r['partial_r']), fmt(r['ci95_low']), fmt(r['ci95_high'])])
    med_file=tables / 'indirect_association.csv'
    if med_file.exists():
        m=pd.read_csv(med_file).iloc[0]
        rows.append(['Fig4', 'ACME', fmt(m['acme']), fmt(m['acme_ci95_low']), fmt(m['acme_ci95_high'])])
        rows.append(['Fig4', 'ADE', fmt(m['ade']), fmt(m['ade_ci95_low']), fmt(m['ade_ci95_high'])])
    pd.DataFrame(rows,columns=['panel','measure','estimate','ci95_low','ci95_high']).to_csv(figures/'ci_summary.csv',index=False)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='./results')
    ap.add_argument('--dpi',type=int,default=300)
    ap.add_argument('--bootstrap',type=int,default=5000)
    args=ap.parse_args()
    tables=Path(args.out)/'tables'
    figures=Path(args.out)/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    values=figure1(tables,figures,args.dpi,args.bootstrap)
    figure2(tables,figures,args.dpi)
    figure3(tables,figures,args.dpi,args.bootstrap)
    figure4(tables,figures,args.dpi)
    write_ci_summary(tables,figures,values)
    print('Saved Fig1 BCDE, Fig2 B, Fig3 ABCDE, Fig4 and ci_summary.csv:',figures)

if __name__=='__main__':
    main()
