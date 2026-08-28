"""
Figure 3 -- the index of operators.

A unit's atlas is a vector over its own channel pairs, so systems with
different channel counts cannot be compared directly. Taking quantiles of each
atom across pairs turns the atlas into a distribution and fixes the length at
9 atoms x 9 quantiles regardless of how many channels the machine has. That is
the class-level object, and it is what makes a motor bench, a pump rig, an air
compressor, a turbofan and a basket of traded assets commensurable at all.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import figstyle as fs        # noqa: E402
import atoms                 # noqa: E402

PRETTY = {'pmsm': 'PMSM motor bench', 'metropt': 'metro air compressor',
          'skab-valve1': 'pump rig, valve1', 'skab-valve2': 'pump rig, valve2',
          'skab-other': 'pump rig, other', 'finance': 'traded assets',
          'cmapss-FD001': 'turbofan FD001', 'cmapss-FD002': 'turbofan FD002',
          'cmapss-FD003': 'turbofan FD003', 'cmapss-FD004': 'turbofan FD004'}


def main():
    fs.setup()
    z = np.load(os.path.join(HERE, 'class_profiles.npz'), allow_pickle=False)
    P, L, QS = z['P'], np.array([str(x) for x in z['L']]), z['QS']
    R = json.load(open(os.path.join(HERE, 'class_results.json')))
    NA, NQ = len(atoms.ATOM_NAMES), len(QS)
    counts = {c: int((L == c).sum()) for c in sorted(set(L))}
    classes = [c for c in sorted(counts) if counts[c] >= 5]
    keep = np.isin(L, classes)
    P, L = P[keep], L[keep]

    fig = plt.figure(figsize=(12.0, 7.6))
    gs = GridSpec(2, 12, figure=fig, hspace=0.62, wspace=5.2,
                  left=0.075, right=0.985, top=0.885, bottom=0.115)

    # --- a class prototypes ---------------------------------------------
    ax = fig.add_subplot(gs[0, :7])
    M = np.stack([P[L == c].mean(0) for c in classes])          # (C, NA*NQ)
    Z = (M - P.mean(0)) / (P.std(0) + 1e-9)
    v = np.nanpercentile(np.abs(Z), 98)
    im = ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-v, vmax=v,
                   interpolation='nearest')
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([PRETTY.get(c, c) for c in classes], fontsize=8)
    ax.set_xticks(np.arange(NA) * NQ + NQ / 2 - 0.5)
    ax.set_xticklabels(atoms.ATOM_NAMES, fontsize=7.5, rotation=35, ha='right')
    for k in range(1, NA):
        ax.axvline(k * NQ - 0.5, color='white', lw=0.8)
    ax.grid(False)
    ax.set_title('a   class prototypes: the atom-quantile profile, 81 numbers '
                 'whatever\n      the channel count', loc='left', fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cb.set_label('sd from pooled mean', fontsize=7)

    # --- b embedding ------------------------------------------------------
    ax = fig.add_subplot(gs[0, 7:])
    Zs = (P - P.mean(0)) / (P.std(0) + 1e-9)
    _, _, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    E = Zs @ Vt[:2].T
    pal = [fs.C['blue'], fs.C['orange'], fs.C['green'], fs.C['red'],
           fs.C['purple'], fs.C['sky'], fs.C['yellow'], fs.C['grey']]
    for k, c in enumerate(classes):
        m = L == c
        ax.scatter(E[m, 0], E[m, 1], s=22, color=pal[k % len(pal)], lw=0.4,
                   edgecolor='white', label=PRETTY.get(c, c))
    ax.set_xlabel('profile PC1'); ax.set_ylabel('profile PC2')
    ax.legend(fontsize=6.8, loc='best', ncol=1)
    ax.set_title('b   classes separate', loc='left', fontsize=9.5)

    # --- c ablation -------------------------------------------------------
    ax = fig.add_subplot(gs[1, :7])
    ab = R.get('ablation', {})
    if ab:
        nms = atoms.ATOM_NAMES
        only = [ab[n]['only'] * 100 for n in nms]
        drop = [ab[n]['without'] * 100 for n in nms]
        x = np.arange(len(nms)); w = 0.38
        ax.bar(x - w / 2, only, width=w, color=fs.C['blue'], label='this atom alone')
        ax.bar(x + w / 2, drop, width=w, color=fs.C['grey'],
               label='all atoms except this one')
        ax.axhline(R['acc'] * 100, color=fs.C['red'], lw=1.2, ls='--',
                   label=f'all atoms ({R["acc"]*100:.1f}%)')
        ax.axhline(R['majority'] * 100, color=fs.C['black'], lw=1.0, ls=':',
                   label=f'majority ({R["majority"]*100:.0f}%)')
        ax.set_xticks(x); ax.set_xticklabels(nms, rotation=35, ha='right',
                                             fontsize=8)
        ax.set_ylabel('class accuracy (%)')
        ax.legend(fontsize=6.8, ncol=2, loc='lower center', framealpha=0.9, frameon=True)
    ax.set_title('c   no single atom carries the class, and dropping the two '
                 'acquisition-sensitive\n      atoms (tau, fill) is the control '
                 'that matters', loc='left', fontsize=9.5)

    # --- d confusion ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 7:])
    cm = np.array(R.get('confusion', []), dtype=float)
    if cm.size:
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        im = ax.imshow(cmn, cmap='magma', vmin=0, vmax=1)
        labs = R.get('confusion_labels', classes)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([PRETTY.get(c, c) for c in labs], rotation=40,
                           ha='right', fontsize=6.5)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels([PRETTY.get(c, c) for c in labs], fontsize=6.5)
        for i in range(len(labs)):
            for j in range(len(labs)):
                if cmn[i, j] > 0.005:
                    ax.text(j, i, f'{cmn[i,j]*100:.0f}', ha='center',
                            va='center', fontsize=6.5,
                            color='white' if cmn[i, j] < 0.55 else 'black')
        ax.grid(False)
    ax.set_title('d   confusion (row-normalised %)', loc='left', fontsize=9.5)

    fig.suptitle('The index of operators: class signatures that compare across '
                 'systems with different channel counts', y=0.955, fontsize=11)
    out = os.path.join(HERE, 'fig3_classes.png')
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.splitext(out)[0] + '.' + ext,
                    dpi=300 if ext == 'png' else None, format=ext)
        print('wrote', os.path.splitext(out)[0] + '.' + ext)
    print('wrote', out)


if __name__ == '__main__':
    main()
