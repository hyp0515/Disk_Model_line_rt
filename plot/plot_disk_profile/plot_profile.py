import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.data import *
import sys
# sys.path.insert(0,'../../')
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup


# print(grid.y)
# print(grid.y.shape)



def plot_polar_mesh(R, Theta, Phi, value, title, colorbar_label, fname, color, type='edge'):
    if type == 'edge':

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Create a pcolormesh plot and add contour lines
        c = ax.pcolormesh(Theta-pi/2, R, value, shading='auto', cmap=color)
        # ax.set_thetamin(0)
        # ax.set_thetamax(90)
        # levels = np.linspace(Z.min(), Z.max(), 4)
        # ax.contour(T-pi/2, R, Z, levels=levels, colors='k', linewidths=.7, linestyles='dashed')

        # # Create a pcolormesh plot and add contour lines    
        c = ax.pcolormesh(Theta+pi/2, R, value, shading='auto', cmap=color)
        # levels = np.linspace(Z.min(), Z.max(), 4)
        # ax.contour(T+pi/2, R, Z, levels=levels, colors='k', linewidths=.7, linestyles='dashed')

        # Add a colorbar with a scale bar at the bottom
        cb = fig.colorbar(c, ax=ax)
        cb.set_label(colorbar_label)
        
        # Hide radial and theta ticks
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(title, pad=20, fontsize=14, color='k')
        plt.savefig(f'./figures/{fname}_edge.png')
    
    if type == 'face':
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

        # Create a pcolormesh plot and add contour lines
        c = ax.pcolormesh(Phi, R, value, shading='auto', cmap=color)
        # ax.set_rmin(np.min(R))
        # ax.set_thetamin(0)
        # ax.set_thetamax(90)
        # levels = np.linspace(Z.min(), Z.max(), 4)
        # ax.contour(T-pi/2, R, Z, levels=levels, colors='k', linewidths=.7, linestyles='dashed')

        # Add a colorbar with a scale bar at the bottom
        cb = fig.colorbar(c, ax=ax)
        cb.set_label(colorbar_label)
        
        # Hide radial and theta ticks
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(title, pad=20, fontsize=14, color='k')
        plt.savefig(f'./figures/{fname}_face.png')


# p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
#                          pancake=False, v_infall=1, mctherm=False, snowline=True, floor=True)
# data = readData(dtemp=True, ddens=True)
# grid = readGrid(wgrid=False)
# abunch3oh = np.where(data.dusttemp[:, :, :, 0]<100, 1e-10, 1e-5)
# factch3oh = abunch3oh/(2.3*mp)
# nch3oh    = 100*data.rhodust[:, :, :, 0]*factch3oh
# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(nch3oh[:, grid.ny//2, :]), 
#                 'Number density map of methanol', r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]',"ndensity_profile_nomc",'BuPu', type='face')
# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(data.dusttemp[:, grid.ny//2, :, 0]), 
#                 'Temperature map', r'log(T) [K]', "T_profile_nomc", 'OrRd', type='face')
# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(data.rhodust[:, grid.ny//2, :, 0]), 
#                 'Dust density map', r'log($\rho$) [g$cm^{-3}$]', "rho_profile_nomc",'BuPu', type='face')

# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(nch3oh[:, :, 0]), 
#                 'Number density map of methanol', r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]',"ndensity_profile_nomc",'BuPu', type='edge')
# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(data.dusttemp[:, :, 0, 0]), 
#                 'Temperature map', r'log(T) [K]', "T_profile_nomc", 'OrRd', type='edge')
# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(data.rhodust[:, :, 0, 0]), 
#                 'Dust density map', r'log($\rho$) [g$cm^{-3}$]', "rho_profile_nomc",'BuPu', type='edge')
# os.system('make cleanall')


# Plot the data
# p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
#                          pancake=False, v_infall=1, mctherm=True, snowline=True, floor=True, combine=True)
# data = readData(dtemp=True, ddens=True)
# grid = readGrid()
# abunch3oh = np.where(data.dusttemp[:, :, :, 0]<100, 1e-10, 1e-5)
# factch3oh = abunch3oh/(2.3*mp)
# nch3oh    = 100*data.rhodust[:, :, :, 0]*factch3oh




# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(nch3oh[:, :, 0]),
#                 'Number density map of methanol', r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]',"ndensity_profile_combine",'BuPu', type='edge')
# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(data.dusttemp[:, :, 0, 0]), 
#                 'Temperature map', r'log(T) [K]', "T_profile_combine", 'OrRd', type='edge')
# plot_polar_mesh(grid.x/au, grid.y, grid.z, np.log10(data.rhodust[:, :, 0, 0]), 
#                 'Dust density map', r'log($\rho$) [g$cm^{-3}$]', "rho_profile_combine",'BuPu', type='edge')

# os.system('make cleanall')

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 12), subplot_kw={'projection': 'polar'})
fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.05)
mins = np.zeros((3))
maxs = np.zeros((3))

p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                         pancake=False, v_infall=1, mctherm=True, snowline=True, floor=True)
data_mc = readData(dtemp=True, ddens=True)
grid_mc = readGrid()

abunch3oh = np.where(data_mc.dusttemp[:, :, :, 0]<100, 1e-10, 1e-5)
factch3oh = abunch3oh/(2.3*mp)
nch3oh_mc    = 100*data_mc.rhodust[:, :, :, 0]*factch3oh
Theta = grid_mc.y
R  =grid_mc.x/au

p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                         pancake=False, v_infall=1, mctherm=False, snowline=True, floor=True)
data_x22 = readData(dtemp=True, ddens=True)
grid_x22 = readGrid()
abunch3oh = np.where(data_x22.dusttemp[:, :, :, 0]<100, 1e-10, 1e-5)
factch3oh = abunch3oh/(2.3*mp)
nch3oh_x22    = 100*data_x22.rhodust[:, :, :, 0]*factch3oh
Theta = grid_x22.y
R  =grid_x22.x/au


mins[0], mins[1], mins[2] = min(np.min(np.log10(data_mc.rhodust[:, :, 0, 0])), np.min(np.log10(data_x22.rhodust[:, :, 0, 0]))), \
                            min(np.min(np.log10(data_mc.dusttemp[:, :, 0, 0])), np.min(np.log10(data_x22.dusttemp[:, :, 0, 0]))), \
                            min(np.min(np.log10(nch3oh_x22[:, :, 0])), np.min(np.log10(nch3oh_x22[:, :, 0])))
maxs[0], maxs[1], maxs[2] = max(np.max(np.log10(data_mc.rhodust[:, :, 0, 0])), np.max(np.log10(data_x22.rhodust[:, :, 0, 0]))), \
                            max(np.max(np.log10(data_mc.dusttemp[:, :, 0, 0])), np.max(np.log10(data_x22.dusttemp[:, :, 0, 0]))), \
                            max(np.max(np.log10(nch3oh_x22[:, :, 0])), np.max(np.log10(nch3oh_x22[:, :, 0])))


cmaps = ['BuPu', 'OrRd', 'BuPu']
titles = ['Dust density', 'Temperature', 'Number density of methanol']
cbar = [r'log($\rho$) [g$cm^{-3}$]', r'log(T) [K]',r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]']
for idx_val, (val_mc, val_x22) in enumerate(zip([data_mc.rhodust[:, :, 0, 0], data_mc.dusttemp[:, :, 0, 0], nch3oh_mc[:, :, 0]], 
                                                [data_x22.rhodust[:, :, 0, 0], data_x22.dusttemp[:, :, 0, 0], nch3oh_x22[:, :, 0]])):
    
    c1 = ax[0, idx_val].pcolormesh(Theta-pi/2, R, np.log10(val_mc), shading='auto', cmap=cmaps[idx_val], vmin=mins[idx_val], vmax=maxs[idx_val])
    ax[0, idx_val].pcolormesh(Theta+pi/2, R, np.log10(val_mc), shading='auto', cmap=cmaps[idx_val], vmin=mins[idx_val], vmax=maxs[idx_val])
    levels = np.linspace(np.log10(val_mc).min(), np.log10(val_mc).max(), 4)
    ax[0, idx_val].contour(Theta-pi/2, R, np.log10(val_mc), levels=levels, colors='k', linewidths=.7, linestyles='dashed')
    ax[0, idx_val].contour(Theta+pi/2, R, np.log10(val_mc), levels=levels, colors='k', linewidths=.7, linestyles='dashed')
    ax[0, idx_val].set_xticks([])
    ax[0, idx_val].set_yticks([])
    if idx_val == 0:
        ax[0, idx_val].set_ylabel('mctherm', fontsize=22, labelpad=40)
        ax[0, idx_val].yaxis.set_label_position("right")
        ax[0, idx_val].yaxis.set_label_coords(-0.1, 0.5)
        

    c2 = ax[1, idx_val].pcolormesh(Theta-pi/2, R, np.log10(val_x22), shading='auto', cmap=cmaps[idx_val], vmin=mins[idx_val], vmax=maxs[idx_val])
    ax[1, idx_val].pcolormesh(Theta+pi/2, R, np.log10(val_x22), shading='auto', cmap=cmaps[idx_val], vmin=mins[idx_val], vmax=maxs[idx_val])
    levels = np.linspace(np.log10(val_x22).min(), np.log10(val_x22).max(), 4)
    ax[1, idx_val].contour(Theta-pi/2, R, np.log10(val_x22), levels=levels, colors='k', linewidths=.7, linestyles='dashed')
    ax[1, idx_val].contour(Theta+pi/2, R, np.log10(val_x22), levels=levels, colors='k', linewidths=.7, linestyles='dashed')
    ax[1, idx_val].set_xticks([])
    ax[1, idx_val].set_yticks([])
    if idx_val == 0:
        ax[1, idx_val].set_ylabel('X22 model', fontsize=22, labelpad=40)
        ax[1, idx_val].yaxis.set_label_position("right")
        ax[1, idx_val].yaxis.set_label_coords(-0.1, 0.5)

    # Add shared colorbar for each column
    fig.colorbar(c1, ax=[ax[0, idx_val], ax[1, idx_val]], orientation='vertical', fraction=0.046*1.5, pad=0.04).set_label(cbar[idx_val], fontsize=18)

    ax[0, idx_val].set_title(titles[idx_val], fontsize=26, color='k')

plt.savefig('profiles.png')

os.system('make cleanall')