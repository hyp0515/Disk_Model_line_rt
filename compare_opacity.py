import numpy as np
import matplotlib.pyplot as plt
import optool
from disk_model import generate_opacity_table

for a in [1, 0.1, 0.01, 0.001]:

    x22_opacity = generate_opacity_table(
        a_min=1e-6, a_max=a,
        q=-3.5, dust_to_gas=0.01
    )

    x22_lam = x22_opacity['lam'] # cm
    x22_kappa_abs = x22_opacity['kappa'] * 100   # 100 is gas-to-dust ratio
    x22_kappa_sca = x22_opacity['kappa_s'] * 100

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9),
                        layout="constrained", gridspec_kw={'wspace': 0.01, 'hspace': 0.05})
    axes = ax.flatten()

    t = ['T<150K', '150K<T<425K', '425K<T<680K', '680K<T<1200K']
    material = ['h2o-w 0.2','astrosil 0.3291', 'fes 0.0743', 'c-org 0.3966']
    for idx in range(len(material)):
        axes[idx].set_title(t[idx])
        axes[idx].plot(x22_lam*10, x22_kappa_abs[idx,:], 'r-', label=r'$\kappa_{abs, X22}$')
        axes[idx].plot(x22_lam*10, x22_kappa_sca[idx,:], 'r:', label=r'$\kappa^{eff}_{sca, X22}$')
        

        p = optool.particle(f'optool -c {' '.join(material[(idx):])} -a 0.01 {a*1e4} 3.5 -l 0.1 10000 101')
        """
        Note:
        Mass fractions don't have to add up to one since they will be renormalized.
        """
        opt_lam = p.lam * 1e-4
        opt_kappa_abs = p.kabs
        opt_kappa_sca = p.ksca * (1- p.gsca)
        axes[idx].plot(opt_lam*10, opt_kappa_abs[0,:], 'g-', label=r'$\kappa_{abs, optool}$')
        axes[idx].plot(opt_lam*10, opt_kappa_sca[0,:], 'g:', label=r'$\kappa^{eff}_{sca, optool}$')

        axes[idx].set_yscale('log')
        axes[idx].set_xscale('log')
        if idx == 2:
            axes[idx].set_ylabel(r'$\kappa$ [cm$^{2}$ g$^{-1}$]')
            axes[idx].set_xlabel(r'$\lambda$ [mm]')
        if idx == 1:
            axes[idx].legend()

    fig.suptitle(f'Compare Opacity ({a*10} mm)')
    plt.savefig(f'compare_kappa_{a*10}mm.pdf', transparent=True)
    plt.close('all')