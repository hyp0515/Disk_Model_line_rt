import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.data import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup

p = problem_setup(a_max=0.01, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, 
                         pancake=False, v_infall=1)

data = readData(dtemp=True, gdens=True, ddens=True)
grid = readGrid()
# print(grid.y)
# print(grid.y.shape)

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.grid(False)
c = ax.pcolormesh(grid.y-pi/2, grid.x/au, np.log10(data.rhodust[:, :, 0, 0]), shading='auto', cmap='jet')
ax.set_thetamin(0)
ax.set_thetamax(90)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Dust density map')
fig.colorbar(c, ax=ax, label=r'log($\rho$) [g$cm^{-3}$]')
plt.savefig('./Figures/profile/rho_profile.png')
plt.close()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.grid(False)
c = ax.pcolormesh(grid.y-pi/2, grid.x/au, np.log10(data.dusttemp[:, :, 0, 0]), shading='auto', cmap='jet')
ax.set_thetamin(0)
ax.set_thetamax(90)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Temperature map')
fig.colorbar(c, ax=ax, label=r'log(T) [K]')
plt.savefig('./Figures/profile/T_profile.png')
plt.close()


abunch3oh = np.where(data.dusttemp[:, :, 0, 0]<100, 1e-10, 1e-5)
factch3oh = abunch3oh/(2.3*mp)
nch3oh    = 100*data.rhodust[:, :, 0, 0]*factch3oh

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.grid(False)
c = ax.pcolormesh(grid.y-pi/2, grid.x/au, np.log10(nch3oh), shading='auto', cmap='jet')
ax.set_thetamin(0)
ax.set_thetamax(90)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Number density map of methanol')
fig.colorbar(c, ax=ax, label=r'log($n_{\mathregular{CH_3OH}}$) [$cm^{-3}$]')
plt.savefig('./Figures/profile/ndensity_profile.png')
plt.close()