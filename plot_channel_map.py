import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup



# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=False)
# os.system(f"radmc3d spectrum incl 0 phi 0 iline 240 widthkms 10 vkms 0 linenlam 100")
# s = readSpectrum('spectrum.out')
# peak_lam = s[np.argmax(s, axis=1),0]
# v_axis = cc*((s[:,0]**2/peak_lam**2-1)/(s[:,0]**2/peak_lam**2+1))*1e-5

# for amax in [0.001, 0.01, 0.1, 1]:
#     problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au)
#     os.system(f"radmc3d spectrum incl 70 phi 0 iline 240 widthkms 5 vkms 0 linenlam 100")
#     s = readSpectrum('spectrum.out')
#     plt.plot(v_axis-5, 1e26*s[:,1], label=f'{amax}cm')
# plt.legend()
# plt.xlabel('Velocity (km/s)')
# plt.ylabel('Intensity (mJy/beam)')
# plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
# plt.savefig('different_amax')



# problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=False)
# os.system(f"radmc3d spectrum incl 0 phi 0 iline 240 widthkms 10 vkms 0 linenlam 100")
# s = readSpectrum('spectrum.out')
# peak_lam = s[np.argmax(s, axis=1),0]
# v_axis = cc*((s[:,0]**2/peak_lam**2-1)/(s[:,0]**2/peak_lam**2+1))*1e-5
# plt.plot(v_axis-5, 1e26*s[:,1], label=f'0'+r'$^\circ$')

# for deg in [30, 60, 90]:
#     # problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au)
#     os.system(f"radmc3d spectrum incl {deg} phi 0 iline 240 widthkms 10 vkms 0 linenlam 100")
#     s = readSpectrum('spectrum.out')
#     plt.plot(v_axis-5, 1e26*s[:,1], label=f'{deg}'+r'$^\circ$')
# plt.legend()
# plt.xlabel('Velocity (km/s)')
# plt.ylabel('Intensity (mJy/beam)')
# plt.title('Spectra of $\mathregular{CH_3OH}$ with different inclination')
# plt.savefig('different_incl.pdf')


# s = readSpectrum('spectrum.out')
# peak_lam = s[np.argmax(s, axis=1),0]
# v_axis = cc*((s[:,0]**2/peak_lam**2-1)/(s[:,0]**2/peak_lam**2+1))*1e-5

# for amax in [0.001, 0.01, 0.1, 1]:
#     problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au)
#     os.system(f"radmc3d spectrum incl 70 phi 0 iline 1 widthkms 5 vkms 0 linenlam 100")
#     s = readSpectrum('spectrum.out')
#     plt.plot(v_axis-5, 1e26*s[:,1], label=f'{amax}cm')
# plt.legend()
# plt.xlabel('Velocity (km/s)')
# plt.ylabel('Intensity (mJy/beam)')
# plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
# plt.savefig('different_amax.pdf')

# makeImage(npix=500,incl=60.,phi=0.,wav=1300.,sizeau=100,posang=142)  
# fig2  = plt.figure()
# a=readImage()
# plotImage(a,log=True,cmap='gist_ncar',au=True,arcsec=False, dpc=140, bunit='snu', maxlog=6)

problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=True)
v_list = np.linspace(-5, 5, 11, endpoint=True)
# os.system("radmc3d image incl 70 iline 1 vkms -5")
# im = readImage('image.out')
# print(im.image.shape)

for v in v_list:
    os.system(f"radmc3d image incl 70 iline 1 vkms {v}")
    im = readImage('image.out')
    # plt.imshow(np.transpose(im.image[:,:,0]),cmap=plt.cm.gist_heat)
    # plt.show()
    plotImage(im, arcsec=True, dpc=140., cmap='hot')