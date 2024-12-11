import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from matplotlib.patches import Ellipse
from scipy import ndimage
import astropy.constants as const
au = const.au.cgs.value
pc = const.pc.cgs.value
import warnings
import sys
sys.path.insert(0,'../')
from example.fit_with_GIdisk.find_center import find_center
filename = '/run/media/hyp0515/fd14f880-ba6f-450f-b82d-98ba3710dc5f/backup/CB68_eDisk/CB68_SBLB_continuum_robust_0.0.image.tt0.fits'
# filename = '/run/media/hyp0515/fd14f880-ba6f-450f-b82d-98ba3710dc5f/backup/CB68/cb68_alma/cb68_setup1_all.rob2.I.image.tt0.pbcor.smooth.dropdeg.fits'

# Load FITS file and data
hdul = fits.open(filename)
data = hdul[0].data  # Assuming the data is in the primary HDU
header = hdul[0].header
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    wcs = WCS(header=header)
hdul.close()

# print(header)
rms_noise = 21e-6
distance_pc = 140  # Distance in parsecs


beam_major = header['BMAJ']
beam_minor = header['BMIN']
beam_pa = header['BPA']
signx = int(-np.sign(header['CDELT1'])) # x propto minus RA
signy = int(np.sign(header['CDELT2'])) 
au_per_pix = abs(header['CDELT1'])/180*np.pi*distance_pc*pc/au 


# # CB68's coordinate
# ra_center  = '16:57:19.64677981' # from 2d gaussian fitting
# dec_center = '-16:09:23.94140037'
# # Convert WCS to pixel coordinates
# coord_center = SkyCoord(ra=ra_center, dec=dec_center, unit=('hourangle', 'deg'))
# ra_center, dec_center = coord_center.ra.degree, coord_center.dec.degree
ra_center, dec_center, _ = find_center(filename, x_lim=[5000, 7000], y_lim=[5000, 7000])

icx_float = (ra_center-header['CRVAL1'])/header['CDELT1']+header['CRPIX1']-1
icy_float = (dec_center-header['CRVAL2'])/header['CDELT2']+header['CRPIX2']-1
icx = int(icx_float)
icy = int(icy_float)

# Crop region
ra_min  = '16:57:19.656'
ra_max  = '16:57:19.63'
dec_min = '-16:09:24.2'
dec_max = '-16:09:23.8'
# Convert WCS to pixel coordinates
coord_min = SkyCoord(ra=ra_min, dec=dec_min, unit=('hourangle', 'deg'))
coord_max = SkyCoord(ra=ra_max, dec=dec_max, unit=('hourangle', 'deg'))
x_min, y_min = skycoord_to_pixel(coord_min, wcs)
x_max, y_max = skycoord_to_pixel(coord_max, wcs)
x_min, x_max = int(x_min), int(x_max)
y_min, y_max = int(y_min), int(y_max)


# Crop the data
cropped_data = data[y_min:y_max, x_min:x_max]
cropped_wcs = wcs.slice((slice(y_min, y_max), slice(x_min, x_max)))

# print(beam_major*3600, beam_minor*3600)

# Convert beam size to pixel scale
beam_major_pixels = (header['BMAJ']/180*np.pi)*distance_pc*pc/au / au_per_pix
beam_minor_pixels = (header['BMIN']/180*np.pi)*distance_pc*pc/au / au_per_pix


# Define contour levels
sigma_levels = [5, 20, 40, 60, 80, 100, 120, 150]
contour_levels = [level * rms_noise for level in sigma_levels]



# Plot the cropped region with AU coordinates
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(projection=cropped_wcs)
im = ax.imshow(cropped_data*1e3, cmap='jet', origin='lower', vmin=0.00, vmax=4)
colorbar = fig.colorbar(im, ax=ax, pad=0.005, aspect=50)
colorbar.set_label('Intensity (mJy/beam)')

# Add contours
contours = ax.contour(cropped_data, levels=contour_levels, colors='w', linewidths=0.8)
# ax.clabel(contours, inline=True, fontsize=8, fmt='%.2e')  # Label contour levels


# Add beam ellipse
beam = Ellipse((115, 10), width=beam_minor_pixels, height=beam_major_pixels,
               angle=beam_pa, edgecolor='w', facecolor='w', lw=1.5, fill=True)
ax.add_patch(beam)

plt.plot([30-10/au_per_pix, 30+10/au_per_pix],[10, 10], "w", lw=3)
plt.text(30, 15, '20 AU', color='w', ha='center', va='top')

ax.set_xlabel('Right Ascension (ICRS)')
ax.set_ylabel('Declination (ICRS)')
plt.tick_params(direction='in', color='w')


plt.savefig('CB68_conti.pdf', transparent=True)
plt.savefig('CB68_conti.png')
plt.close('all')


# cropped_data = data[y_bottom:y_top, x_left:x_right]
# img_size_au = 100
# Npix_half = int(np.ceil(img_size_au/au_per_pix))
# cropped_data = data[icy-Npix_half*signy:icy+(Npix_half+1)*signy:signy, icx-Npix_half*signx:icx+(Npix_half+1)*signx:signx]
# cropped_data = ndimage.shift(cropped_data,
#                                  ((icy-icy_float)*signy, (icx-icx_float)*signx),
#                                  order=1)


# # Plot the cropped region with AU coordinates
# fig = plt.figure(figsize=(8, 6))
# ax = plt.subplot()
# im = ax.imshow(cropped_data, cmap='Oranges', origin='lower', vmin=0.00, vmax=0.06)
# colorbar = fig.colorbar(im, ax=ax, pad=0.005, aspect=50)
# colorbar.set_label('Intensity (Jy/beam)')
# ax.set_yticks([])
# ax.set_xticks([])
# # Add contours
# contours = ax.contour(cropped_data, levels=contour_levels, colors='black', linewidths=0.8)
# # ax.clabel(contours, inline=True, fontsize=8, fmt='%.2e')  # Label contour levels


# # Add beam ellipse
# beam = Ellipse((10, 10), width=beam_minor_pixels, height=beam_major_pixels,
#                angle=beam_pa, edgecolor='black', facecolor='black', lw=1.5, fill=True)
# ax.add_patch(beam)


# plt.plot([60-100/au_per_pix, 60+100/au_per_pix],[75, 75], "k", lw=3)
# plt.text(60, 80, '200 AU', color='k', ha='center', va='top')

# ax.set_xlabel('Right Ascension (ICRS)')
# ax.set_ylabel('Declination (ICRS)')
# plt.tick_params(direction='in', color='black')


# plt.savefig('CB68_conti_crop.pdf', transparent=True)
# plt.close('all')
# print(cropped_data.shape)
# np.save('CB68_conti.npy', cropped_data)


# print(np.unravel_index(np.argmax(cropped_data), cropped_data.shape))