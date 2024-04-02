import numpy as np
from matplotlib import pyplot as plt
from disk_model import *
from problem_setup import problem_setup
from matplotlib.colors import LogNorm

p = problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=True)
r_grid = np.array(p.r_sph)
theta_grid = np.array(p.theta_sph)-0.5*np.pi

v_array = np.loadtxt('gas_velocity.inp', skiprows=2)
v_value_0 = v_array[::200, 2]
v_value_pi = v_array[100::200, 2]

v_map_0 = v_value_0.reshape(54, 399)
v_map_pi = v_value_0.reshape(54, 399)

Theta_0, R_0 = np.meshgrid(theta_grid, r_grid)
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
c = ax.pcolormesh(Theta_0, R_0, v_map_0, cmap='Reds', norm=LogNorm())

Theta_pi, R_pi = np.meshgrid(theta_grid+np.pi, r_grid)
ax.pcolormesh(Theta_pi, R_pi, v_map_pi, cmap='Blues', norm=c.norm)

# # Add contour lines over the colormesh
# contour_levels = np.logspace(np.log10(v_map_0.min()), np.log10(v_map_0.max()), num=5)  # Adjust the levels as needed
# contours = ax.contour(Theta_0, R_0, v_map_0, levels=contour_levels, colors='black')

# contour_levels = np.logspace(np.log10(v_map_pi.min()), np.log10(v_map_pi.max()), num=5)  # Adjust the levels as needed
# contours = ax.contour(Theta_pi, R_pi, v_map_pi, levels=contour_levels, colors='black')

fig.colorbar(c, ax=ax)
plt.show()