"""One unit of work: prep a dataset, or train a single (arm, seed).

Kept as a separate process so the sweep can be fanned out across cores with
xargs -P without any shared mutable state.
"""

import argparse
import json
import os
import time

import torch

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
DATA = os.path.join(os.path.dirname(__file__), '..', 'data')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='run', choices=['prep', 'run'])
    ap.add_argument('--dataset', default='pmsm')
    ap.add_argument('--arm', default='raw')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--experts', type=int, default=6)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--expert-kind', default='mlp')
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--subsample', type=int, default=4)
    ap.add_argument('--n-exact', type=int, default=20000)
    ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float32)
    import runner

    prep_path = os.path.join(DATA, f'prep_{a.dataset}_k{a.k}.pt')

    if a.mode == 'prep':
        t0 = time.time()
        kw = dict(subsample=a.subsample) if a.dataset == 'pmsm' else {}
        blob = runner.prep(DATA, dataset=a.dataset, k=a.k, n_exact=a.n_exact,
                           out=prep_path, **kw)
        print(json.dumps(dict(fidelity=blob['fidelity'],
                              front_secs=blob['front_secs'],
                              n_train=int(blob['Xn']['train'].shape[0]),
                              n_feat=int(blob['Xn']['train'].shape[1]),
                              secs=time.time() - t0), indent=2))
        return

    cfg = dict(k=a.k, epochs=a.epochs, experts=a.experts, width=a.width,
               expert_kind=a.expert_kind, tau=a.tau, bs=a.bs, lr=a.lr)
    blob = torch.load(prep_path, weights_only=False)
    r = runner.run_one(DATA, a.dataset, a.arm, a.seed, cfg, blob=blob)
    tag = a.tag or a.expert_kind
    d = os.path.join(RES, 'jobs')
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f'{a.dataset}_{tag}_k{a.k}_{a.arm}_{a.seed}.json'), 'w') as f:
        json.dump(r, f, indent=2)
    print(f'{a.dataset} {a.arm:10s} seed {a.seed} '
          f'test {r["test"]["mse_mean"]:9.4f}  val {r["val"]["mse_mean"]:9.4f} '
          f'({r["secs"]:.0f}s)')


if __name__ == '__main__':
    main()
