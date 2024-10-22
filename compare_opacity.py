import numpy as np
import warnings
import matplotlib.pyplot as plt
import optool
from disk_model import generate_opacity_table,generate_disk_property_table

for a in [1, 0.1, 0.01, 0.001]:
    print("Calcuatle X22 opacity table \n")
    x22_opacity = generate_opacity_table(
        a_min=1e-6, a_max=a,
        q=-3.5, dust_to_gas=0.01
    )
    x22_lam = x22_opacity['lam'] # cm
    x22_kappa_abs = x22_opacity['kappa']   # 100 is gas-to-dust ratio
    x22_kappa_sca = x22_opacity['kappa_s']

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9),
                        layout="constrained", gridspec_kw={'wspace': 0.01, 'hspace': 0.05})
    axes = ax.flatten()

    t = ['T<150K', '150K<T<425K', '425K<T<680K', '680K<T<1200K']
    fraction = [0.2, 0.3966, 0.0743, 0.3291]
    material = ['h2o-w 0.2', 'c-org 0.3966', 'fes 0.0743', 'astrosil 0.3291'] # species fraction
    for idx in range(len(material)):
        axes[idx].set_title(t[idx])
        axes[idx].plot(x22_lam*10, x22_kappa_abs[idx,:]*100/np.sum(fraction[idx:]), 'r-', label=r'$\kappa_{abs, X22}$')
        axes[idx].plot(x22_lam*10, x22_kappa_sca[idx,:]*100/np.sum(fraction[idx:]), 'r:', label=r'$\kappa^{eff}_{sca, X22}$')
        
        print(f"Calcuatle opacity table from Optool ({t[idx]})")
        p = optool.particle(f'optool -c {' '.join(material[(idx):])} -a 0.01 {a*1e4} 3.5 -l 0.1 10000 101 -mie')
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

    fig.suptitle(f'Compare Opacity (amax = {a*10} mm)')
    plt.savefig(f'compare_kappa_{a*10}mm.pdf', transparent=True)
    plt.close('all')

#########################################################################################

# x22_opacity = generate_opacity_table(
#     a_min=1e-6, a_max=0.001,
#     q=-3.5, dust_to_gas=0.01
# )
# x22_T_grid = x22_opacity['T']
# x22_kappa_p = x22_opacity['kappa_p']
# x22_kappa_r = x22_opacity['kappa_r']
# plt.plot(x22_T_grid, x22_kappa_p, 'r-')
# plt.plot(x22_T_grid, x22_kappa_r, 'r:')

# T_crit = [150, 425, 680, 1200]
# material = ['h2o-w 0.2', 'c-org 0.3966', 'fes 0.0743', 'astrosil 0.3291']
# fraction = [0.2, 0.3966, 0.0743, 0.3291]
# kappa_p = []
# kappa_r = []
# for idx in range(len(material)):
#     p = optool.particle(f'optool -c {' '.join(material[(idx):])} -a 0.01 10 3.5 -l 0.1 10000 101 -mie')
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         p.computemean(tmin=20, tmax=2000, ntemp=100)
#     kappa_p.append(p.kplanck[0,:]*np.sum(fraction[idx:])/100)
#     kappa_r.append(p.kross[0,:]*np.sum(fraction[idx:])/100)
    
# T_grid = p.temp

# kappa_p = np.array(kappa_p)
# kappa_r = np.array(kappa_r)


# def combine_temperature_regimes(y):
#     if y.ndim>2:
#         return np.array([combine_temperature_regimes(y1) for y1 in y])
#     y_out = np.zeros(y.shape[1:])
#     for i in range(len(material))[::-1]:
#         y_out[T_grid<T_crit[i]] = y[i,T_grid<T_crit[i]]
#     return y_out
# kappa_p = combine_temperature_regimes(kappa_p)
# kappa_r = combine_temperature_regimes(kappa_r)

# plt.plot(T_grid, kappa_p, 'g-')
# plt.plot(T_grid, kappa_r, 'g:')
# plt.xlim((20, 1500))
# plt.xscale('log')
# plt.yscale('log')
# plt.show()



# opt_opacity_table = {}
# opt_opacity_table['T_crit'] = T_crit
# opt_opacity_table['T'] = T_grid
# opt_opacity_table['kappa_p'] = kappa_p * 0.01
# opt_opacity_table['kappa_r'] = kappa_r * 0.01


# disk_property_table = generate_disk_property_table(opacity_table=opt_opacity_table)
