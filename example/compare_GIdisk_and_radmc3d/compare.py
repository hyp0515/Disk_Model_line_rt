import numpy
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,'../../')
from disk_model import *


opacity_x22 = generate_opacity_table_1(
    a_min=1e-6, a_max=.01, # min/max grain size
    q=-3.5, # slope for dust size distribution, dn/da ~ a^q
    dust_to_gas=0.01 # dust-to-gas ratio before sublimation
)

kappa_x22 = opacity_x22['kappa']
kappa_s_x22 = opacity_x22['kappa_s']
lam = opacity_x22['lam']

opacity_opt = generate_opacity_table_2(
    a_min=1e-6, a_max=.01, # min/max grain size
    q=-3.5, # slope for dust size distribution, dn/da ~ a^q
    dust_to_gas=0.01 # dust-to-gas ratio before sublimation
)

kappa_opt = opacity_opt['kappa']
kappa_s_opt = opacity_opt['kappa_s']



plt.plot(lam, kappa_s_x22[3,:], 'r')
plt.plot(lam, kappa_s_opt[3,:], 'g')
plt.xscale('log')
plt.yscale('log')
plt.show()