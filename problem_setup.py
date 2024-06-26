import numpy as np
from radmc3dPy.analyze import *
#
# Some natural constants
#
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rs  = 6.96e10        # Solar radius            [cm]
#
# Monte Carlo parameters
#
nphot    = 1000000  # Depend on computer's efficiency
#
# Disk Model
#
from disk_model import *
from vertical_profile_class import DiskModel_vertical

class problem_setup:
    def __init__(self, a_max, Mass_of_star, Accretion_rate, Radius_of_disk, v_infall, 
                 pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False, Rcb=None,
                 abundance_enhancement=1e-5, gas_inside_rcb=False):
        """
        pancake  : simple slab model
        mctherm  : temperature calculated by radmc3d (stellar heating)
        snowline : enhancement of abundance
        floor    : set a floor of rho value to define the boundary of the disk
        combine  : combine accretion and irradiation heating
        Rcb      : Centrifugal Barrier (if not 'None', there is no gas inside Rcb)
        """
        #
        # Write the radmc3d.inp control file
        #
        with open('radmc3d.inp','w+') as f:
            f.write('nphot = %d\n'%(nphot))
            f.write('scattering_mode_max = 2\n')   # Put this to 1 for isotropic scattering
            f.write('istar_sphere = 1\n')
            f.write('setthreads = 14\n') # Depending on the number of cores in the computer
        
        #
        # Write the lines.inp control file
        #
        with open('lines.inp','w+') as f:
            f.write('1\n')
            f.write('1\n')
            f.write('ch3oh    leiden    0    0\n')
                
        #
        # Write the wavelength_micron.inp file
        #
        lam1     = 0.1e0
        lam2     = 1.0e2
        lam3     = 5.0e3
        lam4     = 1.0e4
        n12      = 100  # this section is quite important when mctherm since it covers the peak of the blackbody emission of the central star.
        n23      = 100  # this section focuses on dust and line emission.
        n34      = 50
        lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
        lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
        lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
        lam      = np.concatenate([lam12,lam23,lam34])
        nlam     = lam.size
        with open('wavelength_micron.inp', 'w+') as f:
            f.write('%d\n'%(nlam))
            for value in lam:
                f.write('%13.6e\n'%(value))
        
        #
        # Star parameters and Write the star.inp file
        #
        mstar    = ms  # This is useless in the current version.
        rstar    = rs
        tstar    = ts
        pstar    = np.array([0.,0.,0.])
        with open('stars.inp','w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n'%(nlam))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
            for value in lam:
                f.write('%13.6e\n'%(value))
            f.write('\n%13.6e\n'%(-tstar))
        
        #
        # Disk Model
        #
        d_g_ratio = 0.01
        opacity_table = generate_opacity_table(a_min=0, a_max=a_max, q=-3.5, dust_to_gas=d_g_ratio)
        disk_property_table = generate_disk_property_table(opacity_table)

        DM = DiskModel_vertical(opacity_table, disk_property_table)
        DM.input_disk_parameter(Mstar=Mass_of_star, Mdot=Accretion_rate,
                                Rd=Radius_of_disk, Q=1.5, N_R=200)
        if pancake is True:
            DM.pancake_model()
        DM.extend_to_spherical(NTheta=200, NPhi=100)
        self.DM = DM

        #
        # Write the grid file
        #
        coordsystem = 150  # 100 <= coordsystem < 200 is spherical
        grid_info   = 0  # advised to set =0
        incl_r      = 1
        incl_theta  = 1
        incl_phi    = 1
        nr          = DM.NR
        ntheta      = 2*DM.NTheta-1 # NTheta is semispherical
        nphi        = DM.NPhi
        with open('amr_grid.inp', "w+") as f:
            f.write('1\n')
            f.write('0\n')
            f.write(str(coordsystem)+'\n')
            f.write(str(grid_info)+'\n')
            f.write('%d %d %d\n'%(incl_r, incl_theta, incl_phi))
            f.write('%d %d %d\n'%(nr, ntheta, nphi))
            for value in DM.r_sph_grid:
                f.write('%13.13e\n'%(value))
            for value in DM.theta_sph_grid:
                f.write('%13.13e\n'%(value))
            for value in DM.phi_sph_grid:
                f.write('%13.13e\n'%(value))
        
        #
        # Dust opacity control file
        #
        with open('dustopac.inp','w+') as f:
            f.write('2                          Format number of this file\n')
            f.write('4                          Nr of dust species\n')
            f.write('============================================================================\n')
            f.write('1                          Way in which this dust species is read\n')
            f.write('0                          0=Thermal grain\n')
            f.write('water                  Extension of name of dustkappa_***.inp file\n')
            f.write('============================================================================\n')
            f.write('1                          Way in which this dust species is read\n')
            f.write('0                          0=Thermal grain\n')
            f.write('silicate                   Extension of name of dustkappa_***.inp file\n')
            f.write('============================================================================\n')
            f.write('1                          Way in which this dust species is read\n')
            f.write('0                          0=Thermal grain\n')
            f.write('troilite                   Extension of name of dustkappa_***.inp file\n')
            f.write('============================================================================\n')
            f.write('1                          Way in which this dust species is read\n')
            f.write('0                          0=Thermal grain\n')
            f.write('refractory_organics        Extension of name of dustkappa_***.inp file\n')
        
        #
        # Write dust opacity files
        #
        nlam      = len(opacity_table['lam'])
        lam       = opacity_table['lam']*1e4     # lam in opacity_table is in cgs while RADMC3D uses micro
        kappa_abs = opacity_table['kappa']
        kappa_sca = opacity_table['kappa_s']
        g         = opacity_table['g']
        for idx, composition in enumerate(['water','silicate','troilite','refractory_organics']):
            with open('dustkappa_'+composition+'.inp', "w+") as f:
                f.write('3\n')
                f.write(str(nlam)+'\n')
                for lam_idx in range(nlam):
                    f.write('%13.6e %13.6e %13.6e %13.6e\n'%(lam[lam_idx],kappa_abs[idx,lam_idx],kappa_sca[idx,lam_idx],g[idx,lam_idx]))
        
        #
        # Write the density file
        #
        nspec       = 4
        mass_frac = np.array([0.2, 0.3291, 0.0743, 0.3966])  # quoted from disk_model
        if floor is True:
            rho_dust = np.where(np.log10(DM.rho_sph)<-18, 1e-18, DM.rho_sph)*d_g_ratio  # setting the boundary of the disk
        elif floor is False:
            rho_dust = DM.rho_sph*d_g_ratio
        with open('dust_density.inp', "w+") as f:
            f.write(str(1)+'\n')
            f.write('%d\n'%(nr*ntheta*nphi))
            f.write(str(nspec)+'\n')
            for i in range(nspec):
                data = mass_frac[i]*rho_dust.ravel(order='F')
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')
            f.write('\n')
        
        #
        # Write velocity field
        #
        if Rcb is None:
            vr      = -v_infall*np.sqrt(G*Mass_of_star/(DM.r_sph*au)) # Infall velocity
            vtheta  = 0  # No vertical movement in this version
            if kep is True:
                vphi    = np.sqrt(G*Mass_of_star/(DM.r_sph*au))    # Keplerian velocity
            elif kep is False:
                vphi    = np.zeros(vphi.shape)
        elif Rcb is not None:  # velocity field in Oya et al. 2014
            rcb_idx = np.searchsorted(DM.r_sph, Rcb)
            vr_kep = np.zeros(DM.r_sph[:rcb_idx].shape)  # no infall compoonent inside the centrifugal barrier
            # vr_kep = +np.sqrt(G*Mass_of_star/(DM.r_sph[:rcb_idx]*au))  # if expanding
            vr_infall = -np.sqrt(G*Mass_of_star/(DM.r_sph[rcb_idx:]*au))
            vr = np.concatenate((vr_kep, vr_infall))
            vtheta = 0
            vphi = np.sqrt(G*Mass_of_star/(DM.r_sph*au))
        with open('gas_velocity.inp','w+') as f:
            f.write('1\n')
            f.write('%d\n'%(nr*ntheta*nphi))
            for idx_phi in range(nphi):
                for idx_theta in range(ntheta):
                    for idx_r in range(nr):
                        f.write('%13.6e %13.6e %13.6e \n'%(vr[idx_r],vtheta,vphi[idx_r]))
        
        #
        # Write temperature profile
        #
        if mctherm is True:  # Irradiation heating calculated by RADMC-3D
            if combine is False:
                if floor is True:
                    os.system('radmc3d mctherm')  # Ignore viscous heating calculated by Xu's disk model
                    d = readData(dtemp=True)
                    T = np.where(np.log10(np.tile(rho_dust[:, :, :, np.newaxis], (1, 1, 1, nspec)))<-20, 5, d.dusttemp)
                    T = np.where(d.dusttemp<5, 5, d.dusttemp)
                elif floor is False:
                    os.system('radmc3d mctherm')
                    d = readData(dtemp=True)
                    T = np.where(d.dusttemp<5, 5, d.dusttemp)
            elif combine is True:  # Combination of two heating mechanisms, irradiation and accretion
                if floor is True:
                    xu_T = np.tile(DM.T_sph[:, :, :, np.newaxis], (1, 1, 1, nspec))
                    os.system('radmc3d mctherm')  # Ignore viscous heating calculated by Xu's disk model
                    d = readData(dtemp=True)
                    T_irr  = d.dusttemp
                    T = np.where(np.log10(np.tile(rho_dust[:, :, :, np.newaxis], (1, 1, 1, nspec)))<-20, 5, (T_irr**4+xu_T**4)**(1/4))
                    T = np.where(T<5, 5, T)  # setting the minimum temperature to maintain consistency and prevernt 0 K

                elif floor is False:
                    xu_T = np.tile(DM.T_sph[:, :, :, np.newaxis], (1, 1, 1, nspec))
                    os.system('radmc3d mctherm')
                    d = readData(dtemp=True)
                    T_irr = d.dusttemp
                    T = (T_irr**4+xu_T**4)**(1/4)
                    T = np.where(T<5, 5, T)
            with open('dust_temperature.dat', "w+") as f:
                f.write('1\n')
                f.write('%d\n'%(nr*ntheta*nphi))
                f.write(str(nspec)+'\n')
                for i in range(nspec):
                    data = T[:, :, :, i].ravel(order='F')
                    data.tofile(f, sep='\n', format="%13.6e")
                    f.write('\n')
                f.write('\n')
            
            T_avg = (T[:, :, :, 0]+T[:, :, :, 1]+T[:, :, :, 2]+T[:, :, :, 3])/4
            with open('gas_temperature.inp', "w+") as f:
                f.write('1\n')
                f.write('%d\n'%(nr*ntheta*nphi))
                f.write(str(nspec)+'\n')
                data = T_avg.ravel(order='F')
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')

        elif mctherm is False:  # Accretion heating calculated by Wenrui's Disk Model
            if floor is True:
                T = np.where(np.log10(rho_dust)<-20, 5, DM.T_sph)
            elif floor is False:
                T = DM.T_sph
            

            with open('dust_temperature.dat', "w+") as f:
                f.write('1\n')
                f.write('%d\n'%(nr*ntheta*nphi))
                f.write(str(nspec)+'\n')
                for _ in range(nspec):
                    data = T.ravel(order='F')
                    data.tofile(f, sep='\n', format="%13.6e")
                    f.write('\n')
                f.write('\n')
            
            T_avg = T
            with open('gas_temperature.inp', "w+") as f:
                f.write('1\n')
                f.write('%d\n'%(nr*ntheta*nphi))
                f.write(str(nspec)+'\n')
                data = T_avg.ravel(order='F')
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')

        #
        # Write molecule abundance profile
        #
        if Rcb is None:  
            if snowline is True:
                abunch3oh = np.where(T_avg<100, 1e-10, abundance_enhancement)
                """
                This is over simplied to determine how the abundance is enhanced.
                Realistic chemical configuration is required.
                """
                rho_gas = rho_dust/d_g_ratio
            elif snowline is False:
                abunch3oh = 1e-10
                rho_gas = rho_dust/d_g_ratio  
        elif Rcb is not None:  # The main assumption of Oya's velocity field is that there is no gas inside the centrifugal barrier.
            rcb_idx = np.searchsorted(DM.r_sph, Rcb)
            if snowline is True:
                abunch3oh = np.where(T_avg <100, 1e-10, abundance_enhancement)
                rho_gas = rho_dust/d_g_ratio
                if gas_inside_rcb is False:
                    rho_gas[:rcb_idx, :, :] = 1e-18
            elif snowline is False:
                abunch3oh = 1e-10
                rho_gas = rho_dust/d_g_ratio
                if gas_inside_rcb is False:
                    rho_gas[:rcb_idx, :, :] = 1e-18
        factch3oh = abunch3oh/(2.3*mp)
        nch3oh    = rho_gas*factch3oh
        with open('numberdens_ch3oh.inp','w+') as f:
            f.write('1\n')                       # Format number
            f.write('%d\n'%(nr*ntheta*nphi))           # Nr of cells
            data = nch3oh.ravel(order='F')          # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')

        return

