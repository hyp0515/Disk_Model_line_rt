# import problem_setup as p
# import numpy as np
# from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
# from matplotlib import cm
from radmc3dPy.image import *
from radmc3dPy.analyze import *

makeImage(npix=500,incl=90.,phi=0.,wav=70.,sizeau=100)   # This calls radmc3d 
fig2  = plt.figure()
a=readImage()
plotImage(a,log=True,au=True,maxlog=5,cmap='hot')

os.system("radmc3d sed incl 0 phi 0")

#
# Plotting it "by hand", the SED as seen at 1 pc distance
#
fig3  = plt.figure()
s     = readSpectrum()
lam   = s[:,0]
nu    = 1e4*cc/lam
fnu   = s[:,1]
nufnu = nu*fnu
plt.plot(lam,nufnu)
plt.xscale('log')
plt.yscale('log')
plt.axis([1e-1, 1e4, 1e-10, 1e-1])
plt.xlabel('$\lambda\; [\mu \mathrm{m}$]')
plt.ylabel('$\\nu F_\\nu \; [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]$')

#
# Use the radmc3dPy.analyze tool set for plotting the SED, 
# this time let's plot nuLnu in units of Lsun
#
fig4  = plt.figure()
plotSpectrum(s,nulnu=True,lsun=True,xlg=True,ylg=False,micron=True)
plt.axis([1e-1,1e4,1e-8, 50])


plt.show()