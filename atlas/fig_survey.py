"""
Figure 1 -- which machines have an operating body, and what predicts it.

The drive regime for each system was recorded before any dimension was measured,
so the ordering in panel b is a prediction that held rather than a grouping
chosen afterwards.
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

NICE = {'cnc_mill': 'CNC mill', 'elevator': 'lift', 'hydraulic_rig':
        'hydraulic rig', 'metro_compressor': 'metro compressor',
        'battery_cycling': 'battery cycling', 'gas_turbine': 'gas turbine',
        'transformer': 'transformer', 'steel_plant': 'steel plant',
        'cmapss_turbofan': 'turbofan (sim)', 'pmsm_bench': 'PMSM bench',
        'wind_farm': 'wind farm', 'pump_rig': 'pump rig'}
REG = ['closed-loop, repeated programme', 'closed-loop, free duty',
       'simulated, free duty', 'open-loop, rich excitation']
COL = {REG[0]: fs.C['green'], REG[1]: fs.C['blue'],
       REG[2]: fs.C['grey'], REG[3]: fs.C['red']}
SHORT = {REG[0]: 'closed loop,\nrepeated programme',
         REG[1]: 'closed loop,\nfree duty',
         REG[2]: 'simulated,\nfree duty',
         REG[3]: 'open loop,\nrich excitation'}


def main():
    fs.setup()
    rows = json.load(open(os.path.join(HERE, 'survey.json')))
    rows.sort(key=lambda r: r['ratio'])

    fig = plt.figure(figsize=(12.2, 5.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.42, left=0.10, right=0.98,
                  top=0.84, bottom=0.30)

    # --- a every system, ranked -------------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    y = np.arange(len(rows))
    cols = [COL.get(r['drive'], fs.C['grey']) for r in rows]
    ax.barh(y, [r['ratio'] for r in rows], color=cols, height=.66)
    for i, r in enumerate(rows):
        ax.text(r['ratio'] + .012, i, f"{r['twonn']:.1f} of {r['live']}",
                va='center', fontsize=7.6)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE.get(r['system'], r['system']) for r in rows],
                       fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(.35, color=fs.C['black'], ls='--', lw=1.1)
    ax.text(.355, len(rows) - .3, 'body   |   cloud', fontsize=7.5)
    ax.set_xlim(0, .88)
    ax.set_xlabel('intrinsic dimension / live channels')
    ax.set_title('a   Which machines have an operating body', loc='left',
                 fontsize=10)
    seen = []
    for r in rows:
        if r['drive'] not in seen:
            seen.append(r['drive'])
    handles = [plt.Rectangle((0, 0), 1, 1, color=COL[g]) for g in REG
               if g in seen]
    ax.legend(handles, [g for g in REG if g in seen], fontsize=7,
              loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
              frameon=False)

    # --- b by drive regime -------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    groups = [g for g in REG if any(r['drive'] == g for r in rows)]
    data = [[r['ratio'] for r in rows if r['drive'] == g] for g in groups]
    x = np.arange(len(groups))
    for i, (g, d) in enumerate(zip(groups, data)):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-.09, .09, len(d)),
                   d, s=42, color=COL[g], lw=.5, edgecolor='white', zorder=3)
        ax.plot([i - .26, i + .26], [np.median(d)] * 2, color=fs.C['black'],
                lw=2, zorder=4)
    ax.axhline(.35, color=fs.C['black'], ls='--', lw=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[g] for g in groups], fontsize=7.2)
    ax.set_ylabel('intrinsic dimension / live channels')
    ax.set_ylim(0, .85)
    ax.set_title('b   How a machine is driven\n     predicts whether it has one',
                 loc='left', fontsize=10)

    fig.suptitle('A controller enforcing a setpoint is a constraint, and '
                 'constraints show up as missing dimensions', y=0.955,
                 fontsize=11)
    for ext in ('png', 'pdf'):
        out = os.path.join(HERE, f'fig8_survey.{ext}')
        fig.savefig(out, dpi=300 if ext == 'png' else None, format=ext)
    print('wrote fig8_survey.png + .pdf')
    for g in groups:
        d = [r['ratio'] for r in rows if r['drive'] == g]
        print(f'  {g:<34} n={len(d)}  median {np.median(d):.2f}')


if __name__ == '__main__':
    main()
