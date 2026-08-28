"""Inject result tables into the manuscript, so no number in the paper is
typed by hand."""

import json
import os

ROOT = os.path.join(os.path.dirname(__file__), '..')
RES = os.path.join(ROOT, 'results')
TEX = os.path.join(ROOT, 'paper', 'ioo.tex')

NAMES = {'mono': 'monolith', 'raw': 'raw state', 'invariant': 'invariants $(R,\\Pen)$',
         'raw+inv': 'raw $+$ invariants', 'naive': '$\\Trr V,\\ \\log\\det V$',
         'activity': 'local activity', 'random': 'random projection'}
ORDER = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']
PRETTY = {'pmsm': 'PMSM (K$^2$)', 'cmapss': 'C-MAPSS (cycles$^2$)',
          'ur5e': 'UR5e', 'panda': 'Panda', 'iiwa14': 'iiwa14'}


def load(p, d=None):
    f = os.path.join(RES, p)
    return json.load(open(f)) if os.path.exists(f) else d


def results_table():
    s = load('summary.json', {})
    if not s:
        return '% no results yet\n'
    ds = [d for d in ['pmsm', 'cmapss', 'ur5e', 'panda', 'iiwa14'] if d in s]
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{PRETTY.get(d, d)}}}' for d in ds)
    lines = [r'\begin{table}[t]', r'\centering', r'\small',
             r'\caption{In-distribution test error by router input. One expert '
             r'bank, one budget, identical expert inputs; only the router input '
             r'differs. Percentages and paired $p$-values are against the '
             r'raw-state router over common seeds.}',
             r'\label{tab:results}',
             r'\begin{tabular}{l' + 'rr' * len(ds) + '}', r'\toprule',
             'router input & ' + head + r' \\',
             ' & ' + ' & '.join(['error & vs raw'] * len(ds)) + r' \\',
             r'\midrule']
    for a in ORDER:
        cells = []
        for d in ds:
            v = s[d]['arms'].get(a)
            if not v:
                cells += ['---', '---']; continue
            e = f'{v["test_mean"]:.1f}\\,$\\pm$\\,{v["test_sd"]:.1f}'
            rel = v.get('rel_vs_raw') or '---'
            p = v.get('p_vs_raw') or ''
            if p and float(p) < 0.05:
                rel = f'\\textbf{{{rel}}}'
            cells += [e, rel.replace('%', '\\%')]
        lines.append(NAMES[a] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def diffusion_table():
    d = load('diffusion_check.json', [])
    if not d:
        return '% no diffusion check yet\n'
    rows = '\n'.join(
        f'{r["dataset"]} & {r["excess_kurtosis"]:.1f} & '
        f'{r["tail_excess_ratio"]:,.0f}$\\times$ & {r["ks_radial"]:.3f} & '
        f'{r["sd"]:.3f} \\\\' for r in d)
    return ('\\begin{table}[t]\n\\centering\\small\n'
            '\\caption{Whitened-increment diagnostics. A correctly specified '
            'diffusion gives excess kurtosis $0$, a $5\\sigma$ ratio of '
            '$1\\times$, small KS distance and whitened s.d.\\ $1$.}\n'
            '\\label{tab:diffusion}\n'
            '\\begin{tabular}{lrrrr}\n\\toprule\n'
            'system & excess kurtosis & $>5\\sigma$ vs Gaussian & KS (radial) '
            '& whitened s.d. \\\\\n\\midrule\n' + rows +
            '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')


def regime_table():
    out = []
    for p in ['ur5e', 'panda', 'iiwa14']:
        r = load(f'regime_{p}.json')
        if not r:
            continue
        rows = '\n'.join(
            f'{k.replace("_", " ")} & {v["ami"][0]:.3f} & {v["bacc"][0]:.3f} \\\\'
            for k, v in r['scores'].items())
        out.append('\\begin{table}[t]\n\\centering\\small\n'
                   f'\\caption{{Regime discovery on the {p.upper()}, at matched '
                   'dimension. The regime labels are never a model input.}}\n'
                   f'\\label{{tab:regime-{p}}}\n'
                   '\\begin{tabular}{lrr}\n\\toprule\ncoordinates & AMI & '
                   'balanced acc. \\\\\n\\midrule\n' + rows +
                   '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    return '\n'.join(out) if out else '% no regime results yet\n'


def main():
    tex = open(TEX, encoding='utf-8').read()
    for key, fn in [('% PLACEHOLDER-RESULTS-TABLE', results_table),
                    ('% PLACEHOLDER-DIFFUSION-TABLE', diffusion_table),
                    ('% PLACEHOLDER-REGIME', regime_table)]:
        if key in tex:
            tex = tex.replace(key, fn())
    out = TEX.replace('.tex', '_filled.tex')
    open(out, 'w', encoding='utf-8').write(tex)
    print('wrote', out)


if __name__ == '__main__':
    main()
