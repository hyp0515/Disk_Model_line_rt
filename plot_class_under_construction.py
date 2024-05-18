import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.data import *
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup

class plot:

    def __init__(self, distance = 140, mctherm=True, single=False):
        self.D = distance
        self.single = single
        self.mctherm = True
        return
    

    def plot_sed(self, plot_nu=True, GHz=True, mjy=True, d=None):
        d = self.D

        s = readSpectrum('spectrum.out')
        lam = s[:, 0]
        fnu = s[:, 1]/(d**2)
        if mjy is True:
            fnu = 1e26*fnu
            plt.ylabel('$ Flux Density \; [mJy]$')
            plt.ylim((1e-2, 1e4))
        else:
            fnu = 1e23*fnu
            plt.ylabel('$ Flux Density \; [Jy]$')
            plt.ylim((1e-5, 1e1))

        if plot_nu is True:
            nu = (1e-2*cc)/(1e-6*lam)
            if GHz is True:
                nu = 1e-9*nu
                plt.xlabel('$\\nu [GHz]$')
            else:
                plt.xlabel('$\\nu [Hz]$')
            fig = plt.plot(nu, fnu)
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim((1e2, 1e3))
        else:
            fig = plt.plot(lam, fnu)
            plt.xlabel('$\\lambda [mm]$')
            plt.yscale('log')
            plt.xscale('log')
            plt.xlim((1e2, 1e6))
        return
    

    def plot_spectra(self):
        
        return
    
    
    def plot_map(self, dust=True, gas=True, scattering=True, single_image=False):
        
        def dust_image():

            return

        def channel_map():

            return
        

        if dust is True and gas is False:   # Dust Image
            dust_image()

        elif dust is True and gas is True:  # Channel Maps
            channel_map()



        return
    
    def plot_profile(self):
        
        return
    
    def plot_pv(self):
        
        return
    