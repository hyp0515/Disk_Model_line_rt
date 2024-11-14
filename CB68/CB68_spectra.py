import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from spectral_cube import SpectralCube, OneDSpectrum


filename = '/run/media/hyp0515/fd14f880-ba6f-450f-b82d-98ba3710dc5f/backup/CB68/CB68-Setup1-cube-products/CB68_218.440GHz_CH3OH_joint_0.5_clean.image.pbcor.common.fits'
cube = SpectralCube.read(filename)
# Define the slice (in this example, y and x range from 700 to 800)
region = cube[:, 735:765, 735:765]
integrated_spectrum = region.sum(axis=(1, 2))  # Sum over y and x axes to get 1D spectrum
spectrum_values = integrated_spectrum.value  # Convert to array
# wavelength = region.spectral_axis  # Get the spectral axis
freq0 = 218.440063 * 1e9
v = 29979245800 / 1e5 * (freq0 - region.spectral_axis.value) / freq0
# print(region.pixels_per_beam)
plt.plot(v, spectrum_values/region.pixels_per_beam)
plt.plot([5, 5], [-1, 1], 'k:')
plt.ylim((-0.02, 0.1))
plt.xlabel("Velocity [km/s]")
plt.ylabel("Integrated Flux [Jy/beam]")
plt.title('Methanol '+r'$(4_{2,3}-3_{1,2}, E)$'+" Spectrum")
plt.savefig('CB68_methanol_spectra.pdf', transparent=True)
plt.show()
plt.close()


fig = region.plot_channel_maps(channels=np.linspace(207, 267, 60, endpoint=True).astype(int).tolist(), nx=15, ny=4, cmap='jet', vmin=0.001, vmax=0.040)
plt.show()
