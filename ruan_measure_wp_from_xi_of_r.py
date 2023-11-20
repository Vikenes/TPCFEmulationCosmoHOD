import numpy as np
import sys
from Corrfunc.theory.wp import wp
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import simps

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
# matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{physics}'
params = {'xtick.top': True, 'ytick.right': True, 'xtick.direction': 'in', 'ytick.direction': 'in'}
plt.rcParams.update(params)

box_size = 2000.0
pos1 = np.load(f'/mn/stornext/d8/data/chengzor/void_abacussummit/data/multi/HOD_LOWZ_gal_pos_cos0_ph0_z0.25.npy')
pimax = 100.0


r_binedge = np.geomspace(0.5, 60, 30)
print(r_binedge)
print(f'{pos1.shape = }')
edges = (r_binedge,)
nthreads = 64
results_wp = wp(
    boxsize=box_size, 
    pimax=pimax, 
    nthreads=nthreads, 
    binfile=r_binedge, 
    X=pos1[:, 0],
    Y=pos1[:, 1],
    Z=pos1[:, 2],
    output_rpavg=True,
)
# r_perp = results_wp['rmin']
r_perp = results_wp['rpavg']
w_p = results_wp['wp']



r_bincentre, xiR, xiR_stddev = np.loadtxt(
    f'/mn/stornext/d8/data/chengzor/void_abacussummit/data/xi-R-gg_LOWZ_cos0_z0.25.dat',
    unpack=True,
)
xiR_func = ius(
    r_bincentre,
    xiR,
)
rpara_integral = np.linspace(0, pimax, int(1000))
dr = rpara_integral[1] - rpara_integral[1]
w_p_fromxiR = 2.0 * simps(
    xiR_func(
        np.sqrt(r_perp.reshape(-1, 1)**2 + rpara_integral.reshape(1, -1)**2)
    ),
    rpara_integral,
    axis=-1,
)



fig = plt.figure(figsize=(5, 5))
gs = gridspec.GridSpec(1, 1,)
ax0 = plt.subplot(gs[0])
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.plot(
    r_perp,
    w_p,
    lw=0,
    marker='o',
    markersize=2,
    label='from sim',
)
ax0.plot(
    r_perp,
    w_p_fromxiR,
    lw=1.0,
    label=r'from $\xi^R_{gg}(r)$'
)
ax0.legend()
ax0.set_xlabel(r'$r_{\bot} / (h^{-1}\mathrm{Mpc})$')
ax0.set_ylabel(r'$r_{\bot} w_{p}(r_{\bot})$')
# plt.savefig(
#     "projected.pdf",
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
plt.show()
