import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.data import *
import sys
sys.path.insert(0,'../../')
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup



# print(grid.y)
# print(grid.y.shape)



def plot_polar_mesh(R, T, Z, title, colorbar_label, fname, color):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Create a pcolormesh plot and add contour lines
    c = ax.pcolormesh(T-pi/2, R, Z, shading='auto', cmap=color)
    levels = np.linspace(Z.min(), Z.max(), 4)
    ax.contour(T-pi/2, R, Z, levels=levels, colors='k', linewidths=.7, linestyles='dashed')

    # Create a pcolormesh plot and add contour lines    
    c = ax.pcolormesh(T+pi/2, R, Z, shading='auto', cmap=color)
    levels = np.linspace(Z.min(), Z.max(), 4)
    ax.contour(T+pi/2, R, Z, levels=levels, colors='k', linewidths=.7, linestyles='dashed')

    # Add a colorbar with a scale bar at the bottom
    cb = fig.colorbar(c, ax=ax)
    cb.set_label(colorbar_label)
    
    # Hide radial and theta ticks
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, pad=20, fontsize=14, color='k')
    plt.savefig(f'./Figures/profile/{fname}.png')

# Plot the data
p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                         pancake=False, v_infall=1, mctherm=True, snowline=True, floor=True)
data = readData(dtemp=True, gdens=True, ddens=True)
grid = readGrid()
abunch3oh = np.where(data.dusttemp[:, :, 0, 0]<100, 1e-10, 1e-5)
factch3oh = abunch3oh/(2.3*mp)
nch3oh    = 100*data.rhodust[:, :, 0, 0]*factch3oh
plot_polar_mesh(grid.x/au, grid.y, np.log10(nch3oh), 'Number density map of methanol', r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]',"ndensity_profile",'BuPu')
plot_polar_mesh(grid.x/au, grid.y, np.log10(data.dusttemp[:, :, 0, 0]), 'Temperature map', r'log(T) [K]', "T_profile", 'magma')
plot_polar_mesh(grid.x/au, grid.y, np.log10(data.rhodust[:, :, 0, 0]), 'Dust density map', r'log($\rho$) [g$cm^{-3}$]', "rho_profile",'BuPu')

os.system('make cleanall')

p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                         pancake=False, v_infall=1, mctherm=False, snowline=True, floor=True)
data = readData(dtemp=True, gdens=True, ddens=True)
grid = readGrid()
abunch3oh = np.where(data.dusttemp[:, :, 0, 0]<100, 1e-10, 1e-5)
factch3oh = abunch3oh/(2.3*mp)
nch3oh    = 100*data.rhodust[:, :, 0, 0]*factch3oh
plot_polar_mesh(grid.x/au, grid.y, np.log10(nch3oh), 'Number density map of methanol', r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]',"ndensity_profile_nomc",'BuPu')
plot_polar_mesh(grid.x/au, grid.y, np.log10(data.dusttemp[:, :, 0, 0]), 'Temperature map', r'log(T) [K]', "T_profile_nomc", 'magma')
plot_polar_mesh(grid.x/au, grid.y, np.log10(data.rhodust[:, :, 0, 0]), 'Dust density map', r'log($\rho$) [g$cm^{-3}$]', "rho_profile_nomc",'BuPu')

os.system('make cleanall')

p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                         pancake=False, v_infall=1, mctherm=False, snowline=True, floor=False)
data = readData(dtemp=True, gdens=True, ddens=True)
grid = readGrid()
abunch3oh = np.where(data.dusttemp[:, :, 0, 0]<100, 1e-10, 1e-5)
factch3oh = abunch3oh/(2.3*mp)
nch3oh    = 100*data.rhodust[:, :, 0, 0]*factch3oh
plot_polar_mesh(grid.x/au, grid.y, np.log10(nch3oh), 'Number density map of methanol', r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]',"ndensity_profile_nomc_nofloor",'BuPu')
plot_polar_mesh(grid.x/au, grid.y, np.log10(data.dusttemp[:, :, 0, 0]), 'Temperature map', r'log(T) [K]', "T_profile_nomc_nofloor", 'magma')
plot_polar_mesh(grid.x/au, grid.y, np.log10(data.rhodust[:, :, 0, 0]), 'Dust density map', r'log($\rho$) [g$cm^{-3}$]', "rho_profile_nomc_nofloor",'BuPu')

os.system('make cleanall')