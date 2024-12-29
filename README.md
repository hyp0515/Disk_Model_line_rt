These are the scripts connecting Wenrui Xu's [GIdisk2obs](https://github.com/wxu26/GIdisk2obs) and [RADMC-3D](https://github.com/dullemond/radmc3d-2.0) to produce synthetic observation of dust continuum, SED, line spectrum, and position-velocity (PV) diagram. Fitting pipelines of dust continuum and sed are also included.

This project is supported by College Student Research Scholarship (113-2813-C-110-005-M) by National Science and Technology Council.

## X22_model
In [X22_model/](X22_model), there are 'GIdisk2obs' (X22_model hereafter) source code.

## radmc
In [radmc/](radmc), there are pipelines to transform X22_model to files which RADMC-3D adopts.

## example
In [example](example), there are several examples to demonstrate how to process these scripts

## CB68
In [CB68](CB68), there are python scripts that extracted neccessary data from CB68's fits files. The data are from two ALMA Large Program ([FAUST](https://doi.org/10.3847/1538-4357/ac77e7) and [eDisk](https://doi.org/10.3847/1538-4357/acdd7a))
