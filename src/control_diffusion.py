"""Control: whiten increments from a diffusion we generated ourselves, with no
jumps. If the diagnostic is sound this must return excess kurtosis near zero."""
import numpy as np, torch
from pipeline import fit_sde
from exp_diffusion_check import report

def main():
    torch.set_default_dtype(torch.float64)
    rng = np.random.RandomState(0); n, k, dt = 120000, 3, 0.01
    z = np.zeros((n, k)); z[0] = rng.randn(k) * 0.3
    def b(x): return np.array([-x[0]+0.3*x[1], -0.7*x[1], -0.5*x[2]+0.2*np.sin(x[0])])
    def L(x):
        s = 0.25*(1+0.5*np.tanh(x[0])); t = 0.3*(1+0.4*x[1]**2/(1+x[1]**2))
        return np.array([[s,0,0],[0.08*np.tanh(x[1]),t,0],[0.05,0.03,0.2]])
    for i in range(1, n):
        z[i] = z[i-1] + b(z[i-1])*dt + L(z[i-1]) @ rng.randn(k)*dt**0.5
    Z = torch.tensor(z[:-1]); DZ = torch.tensor(np.diff(z, axis=0))
    sde = fit_sde(Z, DZ, dt, epochs=40, seed=0)
    with torch.no_grad():
        Lc = sde.L(Z)*dt**0.5
        r = (DZ - sde.b(Z)*dt).unsqueeze(-1)
        e = torch.linalg.solve_triangular(Lc, r, upper=False).squeeze(-1).numpy()
    print('control: a true diffusion with no jumps')
    report('control', e)

    # and the same process with 0.3% of steps replaced by jumps
    zj = z.copy()
    jump = rng.rand(n) < 0.003
    zj[jump] += rng.randn(jump.sum(), k) * 2.0
    Zj = torch.tensor(zj[:-1]); DZj = torch.tensor(np.diff(zj, axis=0))
    sdej = fit_sde(Zj, DZj, dt, epochs=40, seed=0)
    with torch.no_grad():
        Lc = sdej.L(Zj)*dt**0.5
        r = (DZj - sdej.b(Zj)*dt).unsqueeze(-1)
        ej = torch.linalg.solve_triangular(Lc, r, upper=False).squeeze(-1).numpy()
    print('positive control: same process, 0.3% of steps are jumps')
    report('jumps', ej)

if __name__ == '__main__':
    main()
