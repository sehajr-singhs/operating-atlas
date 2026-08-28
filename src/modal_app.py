"""
Modal fan-out for the IOO experiments.

The unsupervised front end (chart + SDE + exact invariants + distilled head) is
fitted once per dataset and cached in a volume, together with every arm's
routing coordinates, so that all arms provably consume identical numbers. The
arms then fan out over (arm, seed).

  modal run modal_app.py::prepare --dataset pmsm
  modal run modal_app.py::sweep   --dataset pmsm --seeds 5
"""

import json
import os
import sys

import modal

app = modal.App('ioo-manifold')

image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.5.1', 'numpy==2.1.3', 'pandas==2.2.3',
                      'scikit-learn==1.5.2', 'scipy==1.14.1')
         .add_local_dir(os.path.dirname(os.path.abspath(__file__)), '/root/src'))

vol = modal.Volume.from_name('ioo-data', create_if_missing=True)
VOL = '/vol'
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')


def _untar():
    """cmapss ships as a tar in the volume; unpack once per container."""
    import tarfile
    d = f'{VOL}/cmapss'
    if not os.path.exists(d) and os.path.exists(f'{VOL}/cmapss.tar'):
        os.makedirs(d, exist_ok=True)
        with tarfile.open(f'{VOL}/cmapss.tar') as t:
            t.extractall(d)


@app.function(image=image, volumes={VOL: vol}, cpu=8.0, memory=32768,
              timeout=60 * 60 * 4)
def prepare(dataset: str = 'pmsm', k: int = 3, n_exact: int = 20000,
            subsample: int = 1):
    sys.path.insert(0, '/root/src')
    _untar()
    import runner
    out = f'{VOL}/prep_{dataset}_k{k}.pt'
    blob = runner.prep(VOL, dataset=dataset, k=k, n_exact=n_exact,
                       out=out, subsample=subsample)
    vol.commit()
    return dict(fidelity=blob['fidelity'], front_secs=blob['front_secs'],
                n_train=int(blob['Xn']['train'].shape[0]),
                n_feat=int(blob['Xn']['train'].shape[1]))


@app.function(image=image, volumes={VOL: vol}, cpu=4.0, memory=16384,
              timeout=60 * 60 * 2)
def run_arm(dataset: str, arm: str, seed: int, cfg: dict):
    sys.path.insert(0, '/root/src')
    import runner
    return runner.run_one(VOL, dataset, arm, seed, cfg)


@app.local_entrypoint()
def prep_entry(dataset: str = 'pmsm', k: int = 3, n_exact: int = 20000,
               subsample: int = 1):
    print(json.dumps(prepare.remote(dataset, k, n_exact, subsample), indent=2))


@app.local_entrypoint()
def sweep(dataset: str = 'pmsm', seeds: int = 5, k: int = 3, epochs: int = 40,
          experts: int = 6, width: int = 64, expert_kind: str = 'mlp',
          tag: str = ''):
    cfg = dict(k=k, epochs=epochs, experts=experts, width=width,
               expert_kind=expert_kind)
    arms = ['mono', 'raw', 'invariant', 'raw+inv', 'naive', 'activity', 'random']
    jobs = [(dataset, a, s, cfg) for s in range(seeds) for a in arms]
    out = list(run_arm.starmap(jobs))
    name = f'{dataset}_{tag or expert_kind}_k{k}.json'
    with open(os.path.join(RES, name), 'w') as f:
        json.dump(out, f, indent=2)
    import statistics as st
    print(f'\n=== {dataset} test MSE ===')
    for a in arms:
        v = [r['test']['mse_mean'] for r in out if r['arm'] == a]
        print(f'  {a:10s} {st.mean(v):9.4f} +- {(st.stdev(v) if len(v)>1 else 0):7.4f}  n={len(v)}')
    print(f'wrote results/{name}')
