import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup


###############################################################################
'''
Spectra with different maximum grain sizes
'''
problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=False)
os.system(f"radmc3d spectrum incl 0 phi 0 iline 240 widthkms 5 vkms 0 linenlam 100")
s = readSpectrum('spectrum.out')
peak_lam = s[np.argmax(s, axis=1),0]
v_axis = cc*((s[:,0]**2/peak_lam**2-1)/(s[:,0]**2/peak_lam**2+1))*1e-5

for amax in [0.001, 0.01, 0.1, 1]:
    problem_setup(a_max=amax, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au)
    os.system(f"radmc3d spectrum incl 70 phi 0 iline 240 widthkms 5 vkms 0 linenlam 100")
    s = readSpectrum('spectrum.out')
    plt.plot(v_axis-5, 1e26*s[:,1], label=f'{amax}cm')
plt.legend()
plt.xlabel('Velocity (km/s)')
plt.ylabel('Intensity (mJy/beam)')
plt.title('Spectra of $\mathregular{CH_3OH}$ with different maximum grain size')
plt.savefig('different_amax')
plt.close()

###############################################################################
'''
Spectra with different inclination
# '''
problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au, pancake=False)
os.system(f"radmc3d spectrum incl 0 phi 0 iline 240 widthkms 5 vkms 0 linenlam 100")
s = readSpectrum('spectrum.out')
peak_lam = s[np.argmax(s, axis=1),0]
v_axis = cc*((s[:,0]**2/peak_lam**2-1)/(s[:,0]**2/peak_lam**2+1))*1e-5
plt.plot(v_axis-5, 1e26*s[:,1], label=f'0'+r'$^\circ$')

for deg in [30, 60, 90]:
    # problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=30*au)
    os.system(f"radmc3d spectrum incl {deg} phi 0 iline 240 widthkms 5 vkms 0 linenlam 100")
    s = readSpectrum('spectrum.out')
    plt.plot(v_axis-5, 1e26*s[:,1], label=f'{deg}'+r'$^\circ$')
plt.legend()
plt.xlabel('Velocity (km/s)')
plt.ylabel('Intensity (mJy/beam)')
plt.title('Spectra of $\mathregular{CH_3OH}$ with different inclination')
plt.savefig('different_incl')
plt.close()

###############################################################################
# Under construction
# def plotter(amax=None, mstar=None, mdot=None, Rd=None, incl=None, v_width=None, line=None, pancake=False):
#     '''
#     Parameters can be input with list

#     amax:    maximum grain size (cm)
#     mstar:   mass of star       (Msun)
#     mdot:    accretion rate     (Msun/yr)
#     Rd:      radius of disk     (au)
#     incl:    inclination angle  (deg)
#     v_width: width of spectra   (km/s)
#     line:    transition level  
#     '''

#     if amax is None:        amax = 0.01 
#     if mstar is None:      mstar = 0.14*Msun
#     if mdot is None:        mdot = mstar*1e-5/yr
#     if Rd is None:            Rd = 30*au
#     if incl is None:        incl = 70
#     if v_width is None:  v_width = 5
#     if line is None:        line = 1

#     # if len(amax)>1 or len(mstar)>1 or len(mdot)>1 or len(Rd)>1:
#     problem_setup(a_max=amax, Mass_of_star=mstar, Accretion_rate=mdot, Radius_of_disk=Rd, pancake=pancake)

#     os.system(f"radmc3d spectrum incl {incl} phi 0 iline {line} widthkms {v_width} vkms 0 linenlam 100")
#     s = readSpectrum('spectrum.out')

#     return

