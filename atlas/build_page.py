"""Assemble the findings page, inlining the figures as data URIs because the
artifact CSP blocks every external host."""

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def uri(name):
    with open(os.path.join(HERE, name), 'rb') as f:
        return 'data:image/jpeg;base64,' + base64.b64encode(f.read()).decode()


R = json.load(open(os.path.join(HERE, 'results_pmsm.json')))
RR = json.load(open(os.path.join(HERE, 'results_real.json')))
C = json.load(open(os.path.join(HERE, 'class_results.json')))
ident = R['identification']
per = R['per_atom']

# controlled fleet experiment (from the Kaggle kernels ioo-shape-level{,panda,iiwa14})
def _sl(path):
    return json.load(open(os.path.join(os.path.expanduser('~'), 'kaggle_kernel',
                                       path)))['per_fleet']
SLF = _sl('shape_level2_out/results.json')
PAND = _sl('shape_level_p_out/results.json')


HTML = open(os.path.join(HERE, 'page_template.html'), encoding='utf-8').read()
HTML = (HTML
        .replace('__FIG1__', uri('fig1_concept_web.jpg'))
        .replace('__FIG2__', uri('fig2_results_web.jpg'))
        .replace('__FIG3__', uri('fig3_classes_web.jpg'))
        .replace('__ATLAS_T1__', f"{ident['atlas, invariant (8)']['top1']:.1f}")
        .replace('__ATLAS9_T1__', f"{ident['full atlas (9)']['top1']:.1f}")
        .replace('__RHO_T1__', f"{ident['rho only']['top1']:.1f}")
        .replace('__MARG_T1__', f"{ident['marginals']['top1']:.1f}")
        .replace('__CHANCE__', f"{ident['chance']['top1']:.1f}")
        .replace('__WARP_ATLAS__', f"{R['invariance']['atlas_warped']['top1']:.1f}")
        .replace('__WARP_MARG__', f"{R['invariance']['marginal_warped']['top1']:.1f}")
        .replace('__LEVY_T1__', f"{per['levy']['top1']:.1f}")
        .replace('__CLASS_ACC__', f"{C['acc']*100:.1f}")
        .replace('__CLASS_SD__', f"{C['sd']*100:.1f}")
        .replace('__CLASS_MAJ__', f"{C['majority']*100:.1f}")
        .replace('__CLASS_NOTF__', f"{C['acc_no_tau_fill']*100:.1f}")
        .replace('__NCLASS__', str(C['n_classes']))
        .replace('__SL_CHANCE__',
                 f"{100*SLF['level_only']['P2_atlas']['chance']:.1f}")
        .replace('__SL_ATLAS_LVL__',
                 f"{100*SLF['level_only']['P2_atlas']['top1']:.1f}")
        .replace('__SL_ATLAS_SHP__',
                 f"{100*SLF['shape_only']['P2_atlas']['top1']:.1f}")
        .replace('__SL_ATLAS_ALL__',
                 f"{100*PAND['all']['P2_atlas']['top1']:.1f}")
        .replace('__SL_AE_CLEAN__',
                 f"{100*PAND['all']['P8']['ae_clean']['top1']:.1f}")
        .replace('__SL_AE_WARP__',
                 f"{100*PAND['all']['P8']['ae_warped']['top1']:.1f}")
        .replace('__REAL7_ACC__', f"{100*RR['R1']['atlas'][0]:.1f}")
        .replace('__REAL7_MARG__', f"{100*RR['R1']['marginal'][0]:.1f}"))

out = os.path.join(HERE, 'findings.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print('wrote', out, os.path.getsize(out) // 1024, 'KB')
