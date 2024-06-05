import numpy as np
from matplotlib import pyplot as plt
from radmc3dPy.image import *
from radmc3dPy.analyze import *
import sys
sys.path.insert(0,'../../')
from disk_model import *
from vertical_profile_class import DiskModel_vertical
from problem_setup import problem_setup


problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=False, snowline=False, floor=True, kep=True, Rcb=None)

for mc in [True, False]:
    for snow in [True, False]:
        for flo in [True, False]:
            for r in [None, 10]:
                problem_setup(a_max=0.1, Mass_of_star=0.14*Msun, Accretion_rate=0.14e-5*Msun/yr, Radius_of_disk=70*au, v_infall=1, 
                            pancake=False, mctherm=mc, snowline=snow, floor=flo, kep=True, Rcb=r)