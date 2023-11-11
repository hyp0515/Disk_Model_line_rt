import numpy as np
from matplotlib import pyplot as plt
from disk_model import *
from vertical_profile_class import DiskModel_vertical

opacity_table = generate_opacity_table(a_min=0, a_max=10, q=-3.5, dust_to_gas=0.01)
disk_property_table = generate_disk_property_table(opacity_table)
DM = DiskModel_vertical(opacity_table, disk_property_table, Mstar=0.5*Msun, Mdot=5e-6*Msun/yr, Rd=50*au, Z_max=50*au, Q=1.5, N_R=250, N_Z=250)
DM.precompute_property(miu=2, factor=1.5)
DM.extend_to_spherical(NTheta=500)

# print(DM.theta_grid[-1])
# print(0.5*np.pi)

print(DM.rho_sph) 