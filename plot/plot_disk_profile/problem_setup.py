import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
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
                 pancake=False, mctherm=True, snowline=True, floor=True, kep=True, combine=False):
        """
        pancake  : simple slab model
        mctherm  : temperature calculated by radmc3d (stellar heating)
        snowline : enhancement of abundance
        floor    : set a floor of rho value to define the boundary of the disk
        """
        #
        # Write the radmc3d.inp control file
        #
        with open('radmc3d.inp','w+') as f:
            f.write('nphot = %d\n'%(nphot))
            f.write('scattering_mode_max = 2\n')   # Put this to 1 for isotropic scattering
            # f.write('iranfreqmode = 1\n')
            f.write('istar_sphere = 1\n')
            f.write('tgas_eq_tdust = 1\n')
            f.write('setthreads = 7\n') # Depending on the number of cores in the computer
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
        lam2     = 7.0e0
        lam3     = 25.e0
        lam4     = 1.0e4
        # lam1     = 3.0e2
        # lam2     = 3.0e3
        # lam3     = 3.0e4
        # lam4     = 3.0e5
        n12      = 20
        n23      = 100
        n34      = 30
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
        opacity_table = generate_opacity_table(a_min=0, a_max=a_max, q=-3.5, dust_to_gas=0.01)
        disk_property_table = generate_disk_property_table(opacity_table)

        DM = DiskModel_vertical(opacity_table, disk_property_table)
        DM.input_disk_parameter(Mstar=Mass_of_star, Mdot=Accretion_rate,
                                Rd=Radius_of_disk, Q=1.5, N_R=200)
        if pancake is True:
            DM.pancake_model()
        DM.extend_to_spherical(NTheta=200, NPhi=200)
        self.DM = DM
        #
        # Write the grid file
        #
        iformat     = 1
        grid_style  = 0
        coordsystem = 150  # 100 <= coordsystem < 200 is spherical
        grid_info   = 0  # advised to set =0
        incl_r      = 1
        incl_theta  = 1
        incl_phi    = 1
        nr          = DM.NR
        ntheta      = 2*DM.NTheta-1 # NTheta is semispherical
        nphi        = DM.NPhi
        with open('amr_grid.inp', "w+") as f:
            f.write(str(iformat)+'\n')
            f.write(str(grid_style)+'\n')
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
            f.write('1                          Nr of dust species\n')
            f.write('============================================================================\n')
            f.write('1                          Way in which this dust species is read\n')
            f.write('0                          0=Thermal grain\n')
            f.write('silicate                   Extension of name of dustkappa_***.inp file\n')
        #
        # Write dust opacity files
        #
        iformat   = 3
        nlam      = len(opacity_table['lam'])
        lam       = opacity_table['lam']*1e4     # lam in opacity_table is in cgs while RADMC3D uses micro
        kappa_abs = opacity_table['kappa']
        kappa_sca = opacity_table['kappa_s']
        g         = opacity_table['g']
        # for idx, composition in enumerate(['water','silicate','troilite','refractory_organics']):
        for idx, composition in enumerate(['silicate']): # for now, only silicate is considered
            with open('dustkappa_'+composition+'.inp', "w+") as f:
                f.write(str(iformat)+'\n')
                f.write(str(nlam)+'\n')
                for lam_idx in range(nlam):
                    f.write('%13.6e %13.6e %13.6e %13.6e\n'%(lam[lam_idx],kappa_abs[idx,lam_idx],kappa_sca[idx,lam_idx],g[idx,lam_idx]))
        #
        # Write the density file
        #
        nspec       = 1
        if floor is True:
            rho = np.where(np.log10(DM.rho_sph)<-18, 1e-18, DM.rho_sph)
        elif floor is False:
            rho = DM.rho_sph
        with open('dust_density.inp', "w+") as f:
            f.write(str(1)+'\n')
            f.write('%d\n'%(nr*ntheta*nphi))
            f.write(str(nspec)+'\n')
            data = 0.01*rho.ravel(order='F')         # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
        #
        # Write velocity field
        #
        iformat = 1
        vr      = 0
        vr      = -v_infall*np.sqrt(G*Mass_of_star/(DM.r_sph*au)) # Infall velocity
        vtheta  = 0
        if kep is True:
            vphi    = np.sqrt(G*Mass_of_star/(DM.r_sph*au))    # Keplerian velocity
        elif kep is False:
            vphi    = np.zeros(vphi.shape)

        with open('gas_velocity.inp','w+') as f:
            f.write(str(iformat)+'\n')
            f.write('%d\n'%(nr*ntheta*nphi))
            for idx_phi in range(nphi):
                for idx_theta in range(ntheta):
                    for idx_r in range(nr):
                        f.write('%13.6e %13.6e %13.6e \n'%(vr[idx_r],vtheta,vphi[idx_r]))
            f.write('\n')

        
        if mctherm is True:
            if combine is False:
                if floor is True:
                    os.system('radmc3d mctherm')  # Ignore viscous heating calculated by Xu's disk model
                    d = readData(dtemp=True, ddens=True)
                    T_read  = d.dusttemp[:, :, :, 0]
                    rho_read = d.rhodust[:, :, :, 0]
                    T = np.where(np.log10(rho_read)<-18, 5, T_read)
                    T = np.where(T<5, 5, T)
                    with open('dust_temperature.dat', "w+") as f:
                        f.write(str(iformat)+'\n')
                        f.write('%d\n'%(nr*ntheta*nphi))
                        f.write(str(nspec)+'\n')
                        data = T.ravel(order='F')         # Create a 1-D view, fortran-style indexing
                        data.tofile(f, sep='\n', format="%13.6e")
                        f.write('\n')
                elif floor is False:
                    os.system('radmc3d mctherm')
                    d = readData(dtemp=True)
                    T_read  = d.dusttemp[:, :, :, 0]
                    T = np.where(T<5, 5, T)
                    with open('dust_temperature.dat', "w+") as f:
                        f.write(str(iformat)+'\n')
                        f.write('%d\n'%(nr*ntheta*nphi))
                        f.write(str(nspec)+'\n')
                        data = T.ravel(order='F')         # Create a 1-D view, fortran-style indexing
                        data.tofile(f, sep='\n', format="%13.6e")
                        f.write('\n')
            elif combine is True:
                if floor is True:
                    xu_T = DM.T_sph
                    
                    os.system('radmc3d mctherm')  # Ignore viscous heating calculated by Xu's disk model
                    d = readData(dtemp=True, ddens=True)
                    T_read  = d.dusttemp[:, :, :, 0]
                    rho_read = d.rhodust[:, :, :, 0]
                    T = np.where(np.log10(rho_read)<-18, 5, (T_read**4+xu_T**4)**(1/4))
                    T = np.where(T<5, 5, T)
                    with open('dust_temperature.dat', "w+") as f:
                        f.write(str(iformat)+'\n')
                        f.write('%d\n'%(nr*ntheta*nphi))
                        f.write(str(nspec)+'\n')
                        data = T.ravel(order='F')         # Create a 1-D view, fortran-style indexing
                        data.tofile(f, sep='\n', format="%13.6e")
                        f.write('\n')
                elif floor is False:
                    xu_T = DM.T_sph

                    os.system('radmc3d mctherm')
                    d = readData(dtemp=True)
                    T_read  = d.dusttemp[:, :, :, 0]
                    T = (T_read**4+xu_T**4)**(1/4)
                    T = np.where(T<5, 5, T)
                    with open('dust_temperature.dat', "w+") as f:
                        f.write(str(iformat)+'\n')
                        f.write('%d\n'%(nr*ntheta*nphi))
                        f.write(str(nspec)+'\n')
                        data = T.ravel(order='F')         # Create a 1-D view, fortran-style indexing
                        data.tofile(f, sep='\n', format="%13.6e")
                        f.write('\n')
        elif mctherm is False:
            if floor is True:
                d = readData(ddens=True)
                rho_read = d.rhodust[:, :, :, 0]
                T = np.where(np.log10(rho_read)<-18, 5, DM.T_sph)
                with open('dust_temperature.dat', "w+") as f:
                    f.write(str(iformat)+'\n')
                    f.write('%d\n'%(nr*ntheta*nphi))
                    f.write(str(nspec)+'\n')
                    data = T.ravel(order='F')         # Create a 1-D view, fortran-style indexing
                    data.tofile(f, sep='\n', format="%13.6e")
                    f.write('\n')
            elif floor is False:
                T = DM.T_sph
                with open('dust_temperature.dat', "w+") as f:
                    f.write(str(iformat)+'\n')
                    f.write('%d\n'%(nr*ntheta*nphi))
                    f.write(str(nspec)+'\n')
                    data = T.ravel(order='F')         # Create a 1-D view, fortran-style indexing
                    data.tofile(f, sep='\n', format="%13.6e")
                    f.write('\n')


        if snowline is True:
            d = readData(dtemp=True, ddens=True)
            T_read  = d.dusttemp[:, :, :, 0] 
            abunch3oh = np.where(T_read<100, 1e-10, 1e-5)
            rho_read = d.rhodust[:, :, :, 0]
            factch3oh = abunch3oh/(2.3*mp)
            nch3oh    = rho_read*factch3oh
            with open('numberdens_ch3oh.inp','w+') as f:
                f.write('1\n')                       # Format number
                f.write('%d\n'%(nr*ntheta*nphi))           # Nr of cells
                data = nch3oh.ravel(order='F')          # Create a 1-D view, fortran-style indexing
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')

        elif snowline is False:
            abunch3oh = 1e-10
            d = readData(ddens=True)
            rho_read = d.rhodust[:, :, :, 0]
            factch3oh = abunch3oh/(2.3*mp)
            nch3oh    = rho_read*factch3oh
            with open('numberdens_ch3oh.inp','w+') as f:
                f.write('1\n')                       # Format number
                f.write('%d\n'%(nr*ntheta*nphi))           # Nr of cells
                data = nch3oh.ravel(order='F')          # Create a 1-D view, fortran-style indexing
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')
        

        return


