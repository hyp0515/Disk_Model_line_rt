import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup



# problem_setup(a_max=1, Mass_of_star=1*Msun, Accretion_rate=1e-5*Msun/yr, Radius_of_disk=50*au)

for degree in [0,15,30,60,90]:
    os.system(f"radmc3d spectrum incl {degree} phi 0 iline 1 widthkms 5 vkms 0 linenlam 100")
    s = readSpectrum('spectrum.out')
    if degree == 0:
        peak_lam = s[np.argmax(s, axis=1),0]
        v_axis = cc*((s[:,0]**2/peak_lam**2-1)/(s[:,0]**2/peak_lam**2+1))*1e-5
    plt.plot(v_axis-5, 1e23*s[:,1], label=f'{degree}')
plt.legend()
# plt.show()
plt.savefig('different_incl')

# makeImage(npix=500,incl=60.,phi=0.,wav=1300.,sizeau=100,posang=142)  
# fig2  = plt.figure()
# a=readImage()
# plotImage(a,log=True,cmap='gist_ncar',au=True,arcsec=False, dpc=140, bunit='snu', maxlog=6)