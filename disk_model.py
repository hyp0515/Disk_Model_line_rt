import numpy as np
import matplotlib.pyplot as plt
import pickle
import warnings
import optool
import os


"""
=======================================================================
Constants
=======================================================================
"""
import astropy.constants as const
c_light = const.c.cgs.value
mp = const.m_p.cgs.value
kB = const.k_B.cgs.value
h = const.h.cgs.value
sigma_SB = const.sigma_sb.cgs.value
G = const.G.cgs.value
au = const.au.cgs.value
pc = const.pc.cgs.value
Msun = const.M_sun.cgs.value
yr = 365*24*3600
pi = np.pi
N_A = const.N_A.cgs.value




"""
=======================================================================
Misc functions
=======================================================================
"""
# These functions are written as classes so that they can be pickled
# inside other classes
import scipy.interpolate
class interp1d_log:
    def __init__(self, x, y):
        self.f_log = scipy.interpolate.interp1d(np.log(x),np.log(y), bounds_error=False, fill_value='extrapolate')
    def __call__(self, x):
        return np.exp(self.f_log(np.log(x)))
# for 2d interp, need z.shape = (len(y), len(x))
class interp2d_semi_log_scalar:
    def __init__(self, x, y, z):
        self.f_log = scipy.interpolate.interp2d(x,np.log(y),np.log(z))
    def __call__(self, x, y):
        return  np.exp(self.f_log(x,np.log(y))[0])
class interp2d_log_vectorized:
    def __init__(self, x, y, z):
        self.f_log = np.vectorize(scipy.interpolate.interp2d(np.log(x),np.log(y),np.log(z)))
    def __call__(self, x, y):
        return  np.exp(self.f_log(np.log(x),np.log(y)))
# planck function
def B(T,nu):
    return 2*h*nu**3/c_light**2 * 1/(np.exp(h*nu/(kB*T))-1)
# generate kappa(T) function at the wavelength set in __init__.
class generate_get_kappa_lam:
    def __init__(self, opacity_table, lam, scattering=False):        
        self.T_crit = opacity_table['T_crit']
        self.N_T_crit = len(self.T_crit)
        self.kappa = np.zeros(self.N_T_crit+1)
        kappa_name = 'kappa_s' if scattering else 'kappa'
        for i in range(self.N_T_crit):
            f = interp1d_log(opacity_table['lam'], opacity_table[kappa_name][i])
            self.kappa[i] = f(lam)
    def __call__(self, T):
        kappa = self.kappa
        kappa_nu = 0
        for i in range(self.N_T_crit):
            kappa_nu = kappa_nu + (T<=self.T_crit[i])*(kappa[i]-kappa[i+1])
        return kappa_nu




"""
=======================================================================
EoS
=======================================================================
"""
# piecewise-linear interpolation based on Kunz & Mouschovias 2009
def get_gamma_scalar(T):
    T_to_cgs = 63.52144902566
    x = T/T_to_cgs
    if (x< 0.15742713923228946 ): y= 1.6666666666666674 ;
    elif (x< 0.7840818852092962 ): y= -0.029510344497522694 *x + 1.6713123957786717 ;
    elif (x< 1.2692464735582838 ): y= -0.30536198503015355 *x + 1.8876026701255741 ;
    elif (x< 2.412452061925896 ): y= -0.137250136180486 *x + 1.6742272988097704 ;
    elif (x< 3.5087932817520255 ): y= 0.00045928128751677744 *x + 1.3420099306924729 ;
    elif (x< 6.321611207800625 ): y= 0.01766887702267927 *x + 1.2816250167952665 ;
    elif (x< 9.19447365850311 ): y= 0.0006180548997382237 *x + 1.3894136850298655 ;
    elif (x< 18.436663403919912 ): y= -0.008279871569026645 *x + 1.4712254355622216 ;
    elif (x< 31.485427846457895 ): y= -0.0033079527177521406 *x + 1.3795598412296695 ;
    else: y= 1.2754075346153901 ;
    return y;
get_gamma = np.vectorize(get_gamma_scalar)
def get_cs_scalar(T):
    T_to_cgs = 63.52144902566
    x = T/T_to_cgs
    if (x< 0.15742713923228946 ): y= 1.6666666666666674 ;
    elif (x< 0.7840818852092962 ): y= -0.029510344497522694 *x + 1.6713123957786717 ;
    elif (x< 1.2692464735582838 ): y= -0.30536198503015355 *x + 1.8876026701255741 ;
    elif (x< 2.412452061925896 ): y= -0.137250136180486 *x + 1.6742272988097704 ;
    elif (x< 3.5087932817520255 ): y= 0.00045928128751677744 *x + 1.3420099306924729 ;
    elif (x< 6.321611207800625 ): y= 0.01766887702267927 *x + 1.2816250167952665 ;
    elif (x< 9.19447365850311 ): y= 0.0006180548997382237 *x + 1.3894136850298655 ;
    elif (x< 18.436663403919912 ): y= -0.008279871569026645 *x + 1.4712254355622216 ;
    elif (x< 31.485427846457895 ): y= -0.0033079527177521406 *x + 1.3795598412296695 ;
    else: y= 1.2754075346153901 ;
    return np.sqrt(y*x)*100*au/(1e3*yr);
get_cs = np.vectorize(get_cs_scalar)
# cs^2 = gamma*p/rho
# u = p/rho/(gamma-1)
def get_u(T):
    gamma = get_gamma(T)
    return get_cs(T)**2/gamma/(gamma-1)




"""
=======================================================================
Opacity
=======================================================================
"""
# X22 original opacity table
# def generate_opacity_table(
#     a_min, a_max, q, dust_to_gas,
#     precomputed_grain_properties_fname='data_for_disk_model/opacity_tables/grain_properties.pkl',
#     T_min=20, T_max=2000, N_T=50,
#     ):
#     """
#     Generate an opacity table for given grain size distribution.

#     Args:
#       a_min, a_max: min/max grain size in grain size distribution
#       q: slope for grain size distribution
#       dust_to_gas: dust-to-gas mass ratio (before sublimation)
#       precomputed_grain_properties_fname:
#         name of the pkl file storing pre-computed grain material properties
#       T_min, T_max: min/max temprature for temprature grid
#       N_T: temperature grid resolution

#     Returns:
#       a dictionary containing the following items:
#         T_crit: sublimation temperatures
#         T: temperature grid
#         lam: wavelength grid
#         kappa: absorption opacity at given lam
#                2d array, first dimension corresponds to temperature range
#         kappa_p: Planck mean opacity
#         kappa_r: Rosseland mean opacity
#         kappa_s, kappa_s_p, kappa_s_r:
#           same as kappa, kappa_p, kappa_r, but for effective scattering opacity
#           (1-g)*kappa_scatter
#     """
#     compute_grain_properties_DSHARP()
#     try:
#         with open('./'+precomputed_grain_properties_fname, "rb") as f:
#             grain_properties = pickle.load(f)
#     except FileNotFoundError:
#         try:
#             with open('../../'+precomputed_grain_properties_fname, "rb") as f:
#                 grain_properties = pickle.load(f)
#         except FileNotFoundError:
#             compute_grain_properties_DSHARP(fname='./'+precomputed_grain_properties_fname)
#             with open('./'+precomputed_grain_properties_fname, "rb") as f:
#                 grain_properties = pickle.load(f)
    



#     a_grid = grain_properties['a_grid']
#     N_a = len(a_grid)
#     lam_grid = grain_properties['lam_grid']
#     T_crit = grain_properties['T_crit']
#     N_composition = len(T_crit)
#     T_grid = np.logspace(np.log10(T_min),np.log10(T_max),N_T)
#     eps = 1e-3
#     T_grid = np.sort(np.concatenate((T_grid, T_crit-eps, T_crit+eps))) # insert points around T_crit
#     N_T = len(T_grid)
#     kappa = grain_properties['kappa'] # get opacity: kappa(temperature_regime, a, lam)
#     kappa_s = grain_properties['kappa_s']*(1-np.array(grain_properties['g'])) # effective kappa for anisotropic scattering
#     # print('kappa shape '+ str(len(kappa)))
#     # print('kappa_s shape '+ str(len(kappa_s)))
#     # compute mean opacities
#     nu = np.outer(c_light/lam_grid, [1])
#     T = T_grid
#     with warnings.catch_warnings():
#             warnings.simplefilter('ignore')
#             exp_nu_T = np.exp(h*nu/(kB*T))    
#     B_times_nu = nu**4 / (exp_nu_T-1) # arbitrary unit
#     B_times_nu_norm = B_times_nu/np.sum(B_times_nu,axis=0)
#     # u = dB/dT
#     u_times_nu = nu**4 / (exp_nu_T-1/exp_nu_T) * h*nu/kB/T**2
#     u_times_nu_norm = u_times_nu/np.sum(u_times_nu,axis=0)
#     kappa_p = [None]*N_composition
#     kappa_r = [None]*N_composition
#     kappa_s_p = [None]*N_composition
#     kappa_s_r = [None]*N_composition
#     for i in range(N_composition):
#         kappa_p[i] = kappa[i].dot(B_times_nu_norm)
#         kappa_r[i] = ((kappa[i]**-1).dot(u_times_nu_norm))**-1
#         kappa_s_p[i] = kappa_s[i].dot(B_times_nu_norm)
#         kappa_s_r[i] = ((kappa_s[i]**-1).dot(u_times_nu_norm))**-1
#         # the summations above assume log-uniform lambda grid
#     # kappa_p/r(temperature_regime, a, T)

#     # weighted average across size distribution
#     ia_min = np.argmin(np.abs(a_grid-a_min))
#     ia_max = np.argmin(np.abs(a_grid-a_max))
#     weight = a_grid**(q+4)
#     weight[:max(0,ia_min-1)] = 0
#     weight[ia_max+1:] = 0
#     weight = weight/np.sum(weight)
#     mass_ratio_after_subl = grain_properties['mass_ratio_after_subl']
#     weight = weight.reshape((1,N_a,1))
#     weight = weight*dust_to_gas*mass_ratio_after_subl.reshape((N_composition,1,1))
#     kappa = np.sum(kappa*weight,axis=1) # kappa(temperature_regime, lam)
#     kappa_s = np.sum(kappa_s*weight,axis=1) # kappa(temperature_regime, lam)
#     kappa_p = np.sum(kappa_p*weight,axis=1) # kappa_p/r(temperature_regime, T)
#     kappa_r = np.sum(kappa_r*weight,axis=1)
#     kappa_s_p = np.sum(kappa_s_p*weight,axis=1) # kappa_p/r(temperature_regime, T)
#     kappa_s_r = np.sum(kappa_s_r*weight,axis=1)
#     g = np.sum(grain_properties['g']*weight,axis=1)

#     # combine different temperature regimes
#     def combine_temperature_regimes(y):
#         if y.ndim>2:
#             return np.array([combine_temperature_regimes(y1) for y1 in y])
#         y_out = np.zeros(y.shape[1:])
#         for i in range(N_composition)[::-1]:
#             y_out[T_grid<T_crit[i]] = y[i,T_grid<T_crit[i]]
#         return y_out
#     kappa_p = combine_temperature_regimes(kappa_p)
#     kappa_r = combine_temperature_regimes(kappa_r)
#     kappa_s_p = combine_temperature_regimes(kappa_s_p)
#     kappa_s_r = combine_temperature_regimes(kappa_s_r)

#     opacity_table = {}
#     opacity_table['T_crit'] = T_crit
#     opacity_table['T'] = T_grid
#     opacity_table['lam'] = lam_grid
#     opacity_table['kappa'] = kappa
#     opacity_table['kappa_s'] = kappa_s
#     opacity_table['kappa_p'] = kappa_p
#     opacity_table['kappa_r'] = kappa_r
#     opacity_table['kappa_s_p'] = kappa_s_p
#     opacity_table['kappa_s_r'] = kappa_s_r
#     opacity_table['g'] = g
#     return opacity_table

# import dsharp_opac # this can be installed from https://github.com/birnstiel/dsharp_opac

# def compute_grain_properties_DSHARP(
#     fname='./data_for_disk_model/opacity_tables/grain_properties.pkl',
#     nang=3, # same as the default value in dsharp_opac
#     ):
#     """
#     Store grain properties computed using the DSHARP opacity model
#     (Birnstiel et al. 2018)
    
#     Args:
#       fname: file name for storing computed grain properties
#       nang: number of angles to compute opacity (passed to dsharp_opac)
#     """
    
    
#     # dust grain compoistion following Birnstiel et al. 2018
#     # the four species:
#     # water, scilicate, troilite, refractory organics
#     N_composition = 4
#     rho_grain = np.array([0.92, 3.30, 4.83, 1.50])
#     mass_frac = np.array([0.2, 0.3291, 0.0743, 0.3966])
#     vol_frac = np.array([0.3642, 0.1670, 0.0258, 0.4430])

#     # sublimation temperature from Pollack et al. 1994
#     T_crit = np.array([150, 425, 680, 1200])

#     diel_constants = [dsharp_opac.diel_warrenbrandt08(),
#                       dsharp_opac.diel_draine2003(species='astrosilicates'),
#                       dsharp_opac.diel_henning('troilite'),
#                       dsharp_opac.diel_henning('organics', refractory=True),
#                      ]

#     species_exists = [[1,1,1,1],
#                       [0,1,1,1],
#                       [0,1,1,0],
#                       [0,1,0,0]]
#     # species_exits[i,j] = species j exists in temperature range i
#     species_exists = np.array(species_exists)
#     rho_grain_eff = np.zeros(N_composition)
#     mass_ratio_after_subl = np.ones(N_composition)
#     mixed_diel_constants = [None]*N_composition
#     for i in range(N_composition):
#         mass_ratio_after_subl[i] = np.sum(mass_frac*species_exists[i])
#         current_vol_frac = vol_frac*species_exists[i]
#         current_vol_frac = current_vol_frac/np.sum(current_vol_frac)
#         rho_grain_eff[i] = np.sum(current_vol_frac*rho_grain)
#         mixed_diel_constants[i] = dsharp_opac.diel_mixed(constants=diel_constants,
#                                   abundances=current_vol_frac,
#                                   rule='Bruggeman')
#         mixed_diel_constants[i] = mixed_diel_constants[i].get_normal_object()

#     # generate grids for grain size and wavelength
#     a_min = 1e-6
#     a_max = 1
#     N_a = 121
#     a_grid = np.logspace(np.log10(a_min),np.log10(a_max),N_a)
#     lam_min = 1e-5 # 1000K=0.0002898cm, choose lam_min << this
#     lam_max = 1 # 10K = 0.02898cm, choose lam_max >> this
#     N_lam = 101
#     lam_grid = np.logspace(np.log10(lam_min),np.log10(lam_max),N_lam)

#     mie_data_package = [None]*N_composition
#     for i in range(N_composition):
#         mie_data_package[i] = dsharp_opac.get_mie_coefficients(
#             a_grid, lam_grid, mixed_diel_constants[i],
#             nang=nang, extrapolate_large_grains=False)

#     kappa   = [None]*N_composition # abroption opacity
#     kappa_s = [None]*N_composition # scattering opacity
#     g       = [None]*N_composition # asymmetry factor
#     for i in range(N_composition):
#         m = 4*np.pi/3 * a_grid**3 * rho_grain_eff[i]
#         kappa_both = dsharp_opac.get_kappa_from_q(
#             a_grid, m,
#             mie_data_package[i]['q_abs'],
#             mie_data_package[i]['q_sca'],
#         )
#         kappa[i] = kappa_both[0]
#         kappa_s[i] = kappa_both[1]
#         g[i] = mie_data_package[i]['g']

#     grain_properties = {}
#     grain_properties['a_grid'] = a_grid
#     grain_properties['lam_grid'] = lam_grid
#     grain_properties['kappa'] = kappa
#     grain_properties['kappa_s'] = kappa_s
#     grain_properties['g'] = g
#     grain_properties['T_crit'] = T_crit
#     grain_properties['mass_ratio_after_subl'] = mass_ratio_after_subl

#     # with open(fname,"wb") as f:
#     #     pickle.dump(grain_properties, f)
    
#     try:
#         with open(fname, "wb") as f:
#             pickle.dump(grain_properties, f)
#     except FileNotFoundError:
#         with open('../../'+fname, "wb") as f:
#             pickle.dump(grain_properties, f)
#     return

# Optool to generate opacity table
def generate_opacity_table(
    a_min, a_max, q, dust_to_gas,
    T_min=20, T_max=2000, N_T=50,
    ):
    """
    Generate an opacity table for given grain size distribution. (using dsharp compsition)

    Args:
      a_min, a_max: min/max grain size in grain size distribution (in cm)
      q: slope for grain size distribution
      dust_to_gas: dust-to-gas mass ratio (before sublimation)
      precomputed_grain_properties_fname:
        name of the pkl file storing pre-computed grain material properties
      T_min, T_max: min/max temprature for temprature grid
      N_T: temperature grid resolution

    Returns:
      a dictionary containing the following items:
        T_crit: sublimation temperatures
        T: temperature grid
        lam: wavelength grid
        kappa: absorption opacity at given lam
               2d array, first dimension corresponds to temperature range
        kappa_p: Planck mean opacity
        kappa_r: Rosseland mean opacity
        kappa_s, kappa_s_p, kappa_s_r:
          same as kappa, kappa_p, kappa_r, but for effective scattering opacity
          (1-g)*kappa_scatter
    """
    
    T_crit = [150, 425, 680, 1200]
    material = ['h2o-w 0.2', 'c-org 0.3966', 'fes 0.0743', 'astrosil 0.3291']
    fraction = [0.2, 0.3966, 0.0743, 0.3291]
    fname = ['temp_1', 'temp_2', 'temp_3', 'temp_4']
    kappa = []
    kappa_s = []
    kappa_p = []
    kappa_r = []
    g       = []
    for idx in range(len(material)):
        p = optool.particle(f'optool -c {' '.join(material[(idx):])} -a 0.01 {a_max*1e4} {-q} -l 0.1 10000 101 -mie -radmc','./opacity_table/',
                            silent=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p.computemean(tmin=T_min, tmax=T_max, ntemp=N_T)
        kappa.append(p.kabs[0,:]*dust_to_gas*np.sum(fraction[idx:]))
        kappa_s.append(p.ksca[0]*dust_to_gas*np.sum(fraction[idx:]))
        kappa_p.append(p.kplanck[0,:]*dust_to_gas*np.sum(fraction[idx:]))
        kappa_r.append(p.kross[0,:]*dust_to_gas*np.sum(fraction[idx:]))
        g.append(p.gsca[0,:])
        os.system(f'cp -r ./opacity_table/dustkappa.inp ./dustkappa_{fname[idx]}.inp')
        os.system('rm -r ./opacity_table')

    T_grid = p.temp
    kappa = np.array(kappa)
    kappa_s = np.array(kappa_s)
    kappa_p = np.array(kappa_p)
    kappa_r = np.array(kappa_r)
    g = np.array(g)
    # combine different temperature regimes
    def combine_temperature_regimes(y):
        if y.ndim>2:
            return np.array([combine_temperature_regimes(y1) for y1 in y])
        y_out = np.zeros(y.shape[1:])
        for i in range(len(material))[::-1]:
            y_out[T_grid<T_crit[i]] = y[i,T_grid<T_crit[i]]
        return y_out

    kappa_p = combine_temperature_regimes(kappa_p)
    kappa_r = combine_temperature_regimes(kappa_r)

    opacity_table = {}
    opacity_table['T_crit'] = T_crit
    opacity_table['T'] = T_grid
    opacity_table['lam'] = p.lam * 1e-4
    opacity_table['kappa'] = kappa
    opacity_table['kappa_s'] = kappa_s
    opacity_table['kappa_p'] = kappa_p
    opacity_table['kappa_r'] = kappa_r
    opacity_table['g'] = g
    return opacity_table

"""
=======================================================================
Solve local disk properties
=======================================================================
"""
def generate_disk_property_table(
    opacity_table, N_T_eff=30, N_Sigma_cs=30, T_eff_min=0.5,
    ):
    """
    Generate a table that maps T_eff and Sigma/cs to other local disk
    properties.

    Args:
      opacity_table: opacity table from generate_opacity_table()
      N_T_eff, N_Sigma_cs: resolution of the grid
      T_eff_min: minimum T_eff (max T_eff is determined automatically)

    Returns:
      a dictionary containing:
        T_eff_grid
        Sigma_cs_l, Sigma_cs_r: min/max Sigma/cs allowed at given T_eff
        x: normalized log(Sigma/cs) grid.
           we map x=[0,1] to the full range of allowed log(Sigma/cs) 
           uniformly.
        Sigma, cs, tau_p_mid, tau_r_mid, T_mid: local disk properties
           for given (T,x)
    """
    T_crit = opacity_table['T_crit']
    T_subl = T_crit[-1] # above this temperature kappa -> 0
    T = opacity_table['T']
    kappa_p = opacity_table['kappa_p']
    kappa_r = opacity_table['kappa_r']
    # ignore the last sublimation point
    kappa_p[kappa_p==0] = kappa_p[kappa_p>0][-1]
    kappa_r[kappa_r==0] = kappa_r[kappa_r>0][-1]
    get_kappa_p = interp1d_log(T, kappa_p)
    get_kappa_r = interp1d_log(T, kappa_r)

    def get_T(T_eff, tau, tau_r_mid, tau_p_mid):
        # Hubeny 1990 Eq 3.11
        T = T_eff * ( 3/4*(tau*(1-tau/(2*tau_r_mid)) + 1/np.sqrt(3) + 1/(3*tau_p_mid)) )**(1/4)
        return T
    def get_tau_from_T(T, T_eff, tau_r_mid, tau_p_mid):
        # returns: (1) tau_R for this T, (2) whether this T is reachable in the given vertical temperature profile
        T_ends = get_T(T_eff, np.array([0,tau_r_mid]), tau_r_mid, tau_p_mid)
        # check whether temperature is reachable
        if T>=T_ends[1]:
            return tau_r_mid, False
        if T<=T_ends[0]:
            return 0, False
        a = -1/(2*tau_r_mid)
        b = 1
        c = 1/np.sqrt(3) + 1/3/(tau_p_mid) - T**4/(3/4*T_eff**4)
        tau = (-b+np.sqrt(b**2-4*a*c)) / (2*a)
        return tau, True
    def get_disk_profile(T_eff, tau_r_mid, tau_p_mid):
        # generate tau grid
        N_tau = 100
        tauf = np.linspace(0,tau_r_mid,N_tau)
        tauf = tauf.tolist()
        # add critical temperatures
        for T_crit_one in T_crit:
            tau_crit, in_range = get_tau_from_T(T_crit_one, T_eff, tau_r_mid, tau_p_mid)
            if in_range:
                tauf.append(tau_crit)
        tauf = np.sort(np.array(tauf))
        tau = (tauf[1:]+tauf[:-1])/2
        dtau = tauf[1:]-tauf[:-1]
        T = get_T(T_eff, tau, tau_r_mid, tau_p_mid)
        T_mid = get_T(T_eff, tau_r_mid, tau_r_mid, tau_p_mid)
        kappa_r = get_kappa_r(T)
        kappa_p = get_kappa_p(T)
        cs = get_cs(T)
        Sigma = 2*np.sum(dtau/kappa_r)
        mean_cs = np.sum(cs*dtau/kappa_r)/np.sum(dtau/kappa_r)
        tau_p_mid_actual = np.sum(dtau*kappa_p/kappa_r)
        tau_p_over_tau_r_actual = tau_p_mid_actual/tau_r_mid
        return Sigma, mean_cs, T_mid, tau_p_over_tau_r_actual
    def solve_disk_profile_with_tau_p(T_eff, tau_p_mid):
        tau_r_mid = tau_p_mid
        err = 1
        err_tol = 1e-8
        n_itr = 0
        n_itr_max = 200
        while n_itr<n_itr_max and err>err_tol:
            n_itr += 1
            res = get_disk_profile(T_eff, tau_r_mid, tau_p_mid)
            tau_r_mid_actual = tau_p_mid / res[-1]
            err = np.abs(tau_r_mid_actual/tau_r_mid - 1)
            tau_r_mid = tau_r_mid_actual
        Sigma, mean_cs, T_mid, tau_p_over_tau_r_actual = get_disk_profile(T_eff, tau_r_mid, tau_p_mid)
        return tau_r_mid, Sigma, mean_cs, T_mid

    # find max possible T_eff
    kappa_p_subl = get_kappa_p(T_subl)
    kappa_r_subl = get_kappa_r(T_subl)
    tau_p = np.sqrt(kappa_p_subl/kappa_r_subl/1.5) # this is because at max T_eff, T=T_subl when dT/dtau = 0
    # get tau_r: solve iteratively
    tau_r = tau_p*1
    err = 1
    while err>1e-12:
        T_eff_4 = T_subl**4/(3/4)/(0.5*tau_r + 1/np.sqrt(3) + 1/(3*tau_p))
        tauf = np.linspace(0,tau_r,101)
        tau = (tauf[1:]+tauf[:-1])/2
        dtau = tauf[1:]-tauf[:-1]
        T = 3/4*T_eff_4*(tau*(1-tau/(2*tau_r)) + 1/np.sqrt(3) + 1/(3*tau_p))
        tau_p_over_tau_r = np.sum(get_kappa_p(T)/get_kappa_r(T)*dtau)/tau_r
        tau_r_new = tau_p/tau_p_over_tau_r
        err = np.abs(tau_r_new/tau_r-1)
        tau_r = tau_r_new
    T_eff_max = T_eff_4**(1/4)
    tau_p_at_T_eff_max = tau_p

    # at given T_eff, do the following:
    # 1. solve disk profile for a range of tau_p
    # 2. get corresponding Sigma/cs, and invert that mapping
    def get_vertical_profile_data_at_T_eff(
        T_eff, N_Sigma_cs = 20,
        dtau = 0.2, # step size in log tau
        N_tau_max = 100, # max n_steps in each direction
        speed_up_factor = 1/50, # increase step size in log tau
        ):

        tau_p_list = []
        tau_r_list = []
        Sigma_list = []
        cs_list = []
        T_mid_list = []
        
        # start at tau_p = tau_p_at_T_eff_max, move left and move right until T_mid > T_subl

        tau_p = tau_p_at_T_eff_max*1
        subl = False
        n_itr = 0
        min_Sigma_cs = np.inf
        while n_itr<N_tau_max and (not subl):
            n_itr += 1
            res = solve_disk_profile_with_tau_p(T_eff, tau_p)
            if (res[1]/res[2])<min_Sigma_cs:
                # only include this point if it probes a lower Sigma/cs.
                # this avoids having a single Sigma/cs mapping to multiple states.
                # generally this selection criterion corresponds to prefering the state with lower T_mid.
                min_Sigma_cs = res[1]/res[2]
                tau_p_list.append(tau_p)
                tau_r_list.append(res[0])
                Sigma_list.append(res[1])
                cs_list.append(res[2])
                T_mid_list.append(res[3])
            subl = res[3]>T_subl
            tau_p = tau_p/(1+dtau)**(1+n_itr*speed_up_factor)
        
        tau_p_list = tau_p_list[::-1]
        tau_r_list = tau_r_list[::-1]
        Sigma_list = Sigma_list[::-1]
        cs_list = cs_list[::-1]
        T_mid_list = T_mid_list[::-1]
        
        tau_p = tau_p_at_T_eff_max*1
        subl = False
        n_itr = 0
        while n_itr<N_tau_max and (not subl):
            tau_p = tau_p*(1+dtau)**(1+n_itr*speed_up_factor)
            n_itr += 1
            res = solve_disk_profile_with_tau_p(T_eff, tau_p)
            tau_p_list.append(tau_p)
            tau_r_list.append(res[0])
            Sigma_list.append(res[1])
            cs_list.append(res[2])
            T_mid_list.append(res[3])
            subl = res[3]>T_subl
        
        tau_p_list = np.array(tau_p_list)
        tau_r_list = np.array(tau_r_list)
        Sigma_list = np.array(Sigma_list)
        cs_list = np.array(cs_list)
        T_mid_list = np.array(T_mid_list)
            
        # find critiical Sigma/cs (when T_mid=T_subl)
        f_Sigma_cs_l = interp1d_log(T_mid_list[:2], (Sigma_list/cs_list)[:2])
        f_Sigma_cs_r = interp1d_log(T_mid_list[-2:], (Sigma_list/cs_list)[-2:])
        Sigma_cs_l = f_Sigma_cs_l(T_subl)
        Sigma_cs_r = f_Sigma_cs_r(T_subl)
        
        # remap data between critical values
        x = np.linspace(0,1,N_Sigma_cs)
        Sigma_cs = Sigma_cs_l * (Sigma_cs_r/Sigma_cs_l)**x
        f_tau_p = interp1d_log(Sigma_list/cs_list, tau_p_list)
        f_tau_r = interp1d_log(Sigma_list/cs_list, tau_r_list)
        f_Sigma = interp1d_log(Sigma_list/cs_list, Sigma_list)
        f_cs    = interp1d_log(Sigma_list/cs_list, cs_list)
        f_T_mid = interp1d_log(Sigma_list/cs_list, T_mid_list)
        tau_p = f_tau_p(Sigma_cs)
        tau_r = f_tau_r(Sigma_cs)
        Sigma = f_Sigma(Sigma_cs)
        cs    = f_cs(Sigma_cs)
        T_mid = f_T_mid(Sigma_cs)

        return [Sigma_cs_l, Sigma_cs_r], [tau_p, tau_r, Sigma, cs, T_mid]

    data_boundary = []
    data_profile = []
    T_eff_grid = np.logspace(np.log10(T_eff_min), np.log10(T_eff_max), N_T_eff)
    for T_eff in T_eff_grid:
        res = get_vertical_profile_data_at_T_eff(T_eff, N_Sigma_cs=N_Sigma_cs)
        data_boundary.append(res[0])
        data_profile.append(res[1])
    data_boundary = np.transpose(np.array(data_boundary)) # l/r, T
    data_profile = np.transpose(np.array(data_profile), (1,0,2)) # variable, T, x
    disk_property_table = {}
    disk_property_table['T_eff_grid'] = T_eff_grid
    disk_property_table['x'] = np.linspace(0,1,data_profile.shape[-1])
    disk_property_table['Sigma_cs_l'] = data_boundary[0]
    disk_property_table['Sigma_cs_r'] = data_boundary[1]
    disk_property_table['tau_p_mid'] = data_profile[0]
    disk_property_table['tau_r_mid'] = data_profile[1]
    disk_property_table['Sigma'] = data_profile[2]
    disk_property_table['cs'] = data_profile[3]
    disk_property_table['T_mid'] = data_profile[4]
    return disk_property_table

"""
=======================================================================
Disk model class
=======================================================================
"""
class DiskModel:
    """
    Parametrized disk model for generating radial porfiles of disk
    properties and flux density at given wavelengths.

    Attributes:
      (all in cgs)
      M: total mass
      Mstar: stellar mass
      Mdot: accretion rate
      Rd: disk size
      Q: Toomre Q
      
      R: radius grid (R[0]=0)
      Sigma, T_mid, tau_p_mid, tau_r_mid: radial profile at R[1:]
      MR: M(<R) profile at R[1:]
    """
    def __init__(self, opacity_table, disk_property_table):
        self.M = 1*Msun
        self.Mstar = 0.5*Msun
        self.Mdot = 1e-5*Msun/yr
        self.Rd = 40*au
        self.Q = 1.5
        self.load_opacity_functions(opacity_table)
        self.load_disk_property_functions(disk_property_table)
        return
    def load_opacity_functions(self, opacity_table):
        self.opacity_table = opacity_table
        T_crit = opacity_table['T_crit']
        self.T_subl = T_crit[-1]
        T = opacity_table['T']
        # lam = opacity_table['lam']
        kappa_p = opacity_table['kappa_p']
        kappa_r = opacity_table['kappa_r']
        # ignore the last sublimation point for kappa_p and kappa_r
        kappa_p[kappa_p==0] = kappa_p[kappa_p>0][-1]
        kappa_r[kappa_r==0] = kappa_r[kappa_r>0][-1]
        self.get_kappa_p = interp1d_log(T, kappa_p)
        self.get_kappa_r = interp1d_log(T, kappa_r)
        return
    def load_disk_property_functions(self, disk_property_table):
        T_eff_grid = disk_property_table['T_eff_grid']
        x = disk_property_table['x']
        self.f_tau_p_mid = interp2d_semi_log_scalar(x,T_eff_grid,disk_property_table['tau_p_mid'])
        self.f_tau_r_mid = interp2d_semi_log_scalar(x,T_eff_grid,disk_property_table['tau_r_mid'])
        self.f_Sigma = interp2d_semi_log_scalar(x,T_eff_grid,disk_property_table['Sigma'])
        self.f_T_mid = interp2d_semi_log_scalar(x,T_eff_grid,disk_property_table['T_mid'])
        self.f_Sigma_cs_l = interp1d_log(T_eff_grid, disk_property_table['Sigma_cs_l'])
        self.f_Sigma_cs_r = interp1d_log(T_eff_grid, disk_property_table['Sigma_cs_r'])
        self.T_eff_max = T_eff_grid[-1]
        return
    def solve_local_disk_properties(self, Omega, kappa):
        Mdot = self.Mdot
        Q = self.Q
        T_eff = (Mdot*Omega**2/(4*pi*sigma_SB))**(1/4)
        if T_eff > self.T_eff_max:
            return 1e-8, 1e-8, 0, 20, self.T_eff_max
        Sigma_cs_l = self.f_Sigma_cs_l(T_eff)
        Sigma_cs_r = self.f_Sigma_cs_r(T_eff)
        Sigma_cs = kappa/(pi*G*Q)
        if Sigma_cs < Sigma_cs_l:
            tau_p_mid, tau_r_mid, Sigma, T_mid = 1e-8, 1e-8, 0, 20
        elif Sigma_cs > Sigma_cs_r:
            tau_p_mid, tau_r_mid, Sigma, T_mid = 1e-8, 1e-8, 0, 20
            #tau_p_mid = self.f_tau_p_mid(1,T_eff)
            #tau_r_mid = self.f_tau_r_mid(1,T_eff)
            #Sigma_dust = self.f_Sigma(1,T_eff)
            #T_mid = self.f_T_mid(1,T_eff)
            #cs_dust = Sigma_dust/Sigma_cs_r
            #cs_dustless = get_cs_scalar(self.T_subl)
            #a = 1
            #b = 2*Sigma_dust - Sigma_cs * cs_dustless
            #c = Sigma_dust**2 - Sigma_cs * cs_dust * Sigma_dust
            #Sigma_dustless = (-b + np.sqrt(b**2-4*a*c))/(2*a)
            #Sigma = Sigma_dustless + Sigma_dust
        else:
            x = np.log(Sigma_cs/Sigma_cs_l) / np.log(Sigma_cs_r/Sigma_cs_l)        
            tau_p_mid = self.f_tau_p_mid(x,T_eff)
            tau_r_mid = self.f_tau_r_mid(x,T_eff)
            Sigma = self.f_Sigma(x,T_eff)
            T_mid = self.f_T_mid(x,T_eff)
        return tau_p_mid, tau_r_mid, Sigma, T_mid, T_eff
    def generate_disk_profile(
        self,Mstar=None,Mdot=None,Rd=None,Q=None,
        N_R = 50, # R grid resolution
        N_itr=8, plot_itr=False,
        ):
        """
        Generate radial disk profile (Sigma, T_mid, tau_p_mid, tau_r_mid)

        Args:
          Mstar, Mdot, Rd, Q: set to None to use current values
                              stored in self
        """
        # update parameters
        if Mstar is not None: self.Mstar = Mstar
        if Mdot is not None: self.Mdot = Mdot
        if Rd is not None: self.Rd = Rd
        if Q is not None: self.Q = Q
        Mstar = self.Mstar
        Mdot = self.Mdot
        Rd = self.Rd
        Q = self.Q
        # set up grid
        Rmin = min(0.05*au, Rd/N_R)
        R = np.concatenate(([0],np.logspace(np.log10(Rmin), np.log10(Rd), N_R)))
        Sigma = np.zeros(N_R+1)
        MR = Mstar * np.ones(N_R)
        tau_p_mid, tau_r_mid, T_mid, T_eff = np.zeros(N_R), np.zeros(N_R), np.zeros(N_R), np.zeros(N_R)
        # iteratively update disk profile
        for n in range(N_itr):
            # mass -> Omega, kappa
            MR = Mstar + np.cumsum(pi*(R[1:]-R[:-1])*(Sigma[1:]*R[1:]+Sigma[:-1]*R[:-1]))
            Omega = np.sqrt(G*MR/R[1:]**3)
            kappa = Omega*np.minimum(2,np.sqrt(1+2*pi*R[1:]*R[1:]*Sigma[1:]/MR))
            # update disk profile
            for i in range(N_R):
                tau_p_mid[i], tau_r_mid[i], Sigma[i+1], T_mid[i], T_eff[i] = \
                self.solve_local_disk_properties(Omega[i], kappa[i])
            if plot_itr:
                plt.plot(R[1:], Sigma[1:])
        self.R = R # N_R+1
        self.MR = MR
        self.M = MR[-1]
        self.Sigma = Sigma[1:]
        self.T_eff = T_eff
        self.T_mid = T_mid
        self.tau_r_mid = tau_r_mid
        self.tau_p_mid = tau_p_mid
        return