import pylab as pl
import numpy as np
from astropy.visualization import quantity_support
from astropy import units as u
from astropy import wcs
import matplotlib.pyplot as plt
pl.rcParams['figure.facecolor']='w'

from spectral_cube import SpectralCube

from pvextractor import extract_pv_slice, Path

from astropy.io import fits

import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# hdu_list = fits.open('/mnt/storage/CB68_setup1/CB68-Setup1-cube-products/CB68_218.440GHz_CH3OH_joint_0.5_clean.image.pbcor.common.fits')
# print(hdu_list.info())


# image_data = fits.getdata('/mnt/storage/CB68_setup1/CB68-Setup1-cube-products/CB68_218.440GHz_CH3OH_joint_0.5_clean.image.pbcor.common.fits')
# print(image_data.shape)

cube = SpectralCube.read('/mnt/storage/CB68_setup1/CB68-Setup1-cube-products/CB68_218.440GHz_CH3OH_joint_0.5_clean.image.pbcor.common.fits')
# print(cube.spectral_axis.value)
freq0 = 218.440063*1e9
v = 29979245800 / 1e5 * (freq0 - cube.spectral_axis.value) / freq0
path = Path([(793, 706), (710, 789)])

pvdiagram = extract_pv_slice(cube=cube, path=path, spacing=1)


# im = plt.imshow(pvdiagram.data[207:267, 30:86], cmap='jet', vmin=0)
# print(pvdiagram.data[207:267, 30:86].shape)
v_axis = v[207:267]
offset = np.linspace(-1, 1, 50, endpoint=True)
O, V = np.meshgrid(offset, v_axis)
fig, ax = plt.subplots()

im = ax.pcolormesh(O, V, pvdiagram.data[207:267, 36:86][:, ::-1], shading='auto', vmin=0.005, cmap='jet')
ax.plot([0, 0], [v_axis[-1], 10], 'w:')
ax.plot([-1, 1], [5, 5], 'w:')
ax.set_yticks([v_axis[-1], 2, 4, 6, 8, v_axis[0]], [0, 2, 4, 6, 8, 10])
ax.set_xticks([-1, -0.5, 0, 0.5, 1],['-1.00"', '-0.50"', '0.00"', '-0.50"', '1.00"'])
ax.set_ylabel('Velocity (km/s)',fontsize = 12)
ax.set_xlabel('Offset (")',fontsize = 12)
ax.text(-0.8,9, 'CB68 '+r'$\mathregular{CH_3OH}}$'+ ' $(4_{2,3}-3_{1,2}E)$', color='white')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Jy/beam',fontsize = 12)
# plt.subplots_adjust(right=0.8)
# cbar.ax.set_position([0.85, 0.15, 0.05, 0.7])
scalebar = AnchoredSizeBar(ax.transData,
                           .7, '100 AU', 'lower right', 
                           pad=0.1,
                           color='white',
                           frameon=False,
                           size_vertical=.1,
                           fontproperties=fm.FontProperties(size=12))
ax.add_artist(scalebar)

plt.savefig('cb68_pv.png')
plt.show()
# 218.440063