import numpy as np
from matplotlib import pyplot as plt
import emcee
import corner
import sys
sys.path.insert(0,'../../')
from disk_model import *
from radmc.setup import *
from multiprocessing import Pool
import time 
from datetime import datetime
import shutil
import h5py

reader = emcee.backends.HDFBackend("progress.h5")


fig, axes = plt.subplots(4, figsize=(10, 10), sharex=True)
samples = reader.get_chain()
label = [r'log($a_{max}$)',r'$M_{*}$', r'log($\dot{M}$)', 'Radius [au]']
for i in range(4):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(label[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
# plt.show()
plt.savefig('chain_step.pdf', transparent = True)
plt.close()

flat_samples = reader.get_chain(flat=True)
fig = corner.corner(
    flat_samples, labels=label,
    show_titles=True, plot_datapoints=True, quantiles=[0.16, 0.5, 0.84])
fig.savefig('posterior.pdf', transparent=True)
plt.close()



observed_Freq = np.array([
                        #   233.8, 233.8,
                          233.8, 233.8, 233.8, 
                        #   246.7,
                          246.7, 246.7, 246.7, 246.7, 246.7,
                        #    95.0,  95.0,  95.0
                           ])
observed_Flux = np.array([   
                        #   91,    87,
                             56,    55,    59,
                            # 101,
                            62,    60,    60,    61,    66,
                            # 6.9,   7.2,   7.6
                            ])
err           = np.array([
                        #   0.730, 0.520,
                          0.054, 0.150, 0.029,
                        #   0.740,
                          0.120, 0.300, 0.220, 0.120, 0.040,
                        #   0.130, 0.040, 0.022
                          ])

model = np.load('record.npz')
nu = model['nu']
fnu = model['fnu']


for i in range(len(nu)):

    plt.plot(nu[i], fnu[i], "C1", alpha=0.005)
plt.errorbar(observed_Freq, observed_Flux, yerr=err, fmt=".k", capsize=0)
plt.xlabel('$\\nu [GHz]$')
plt.ylabel('Flux density [mJy]')
plt.savefig('plot.pdf', transparent=True)
plt.close()
