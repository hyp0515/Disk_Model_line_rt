import os
import numpy as np
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from radmc3dPy.data import *

class generate_simulation:
    
    def __init__(self, parms,
                 channel=True,
                 pv=True,
                 conti=True,
                 sed=False,
                 line_spectrum=False
                 ):
        
        self.save_out = getattr(parms, 'save_out', True)
        self.save_npz = getattr(parms, 'save_npz', True)
        if (self.save_out is False) and (self.save_npz is False):
            self.save_out = True
        
        if channel is True:
            try:
                self.generate_cube(parms, parms.channel_cube_parms)
                print('Image cube is generated')
            except:
                print('Wrong parameters to generate image cube')
        
        if pv is True:
            try:
                self.generate_cube(parms, parms.pv_cube_parms)
                print('Image cube is generated')
            except:
                print('Wrong parameters to generate image cube')
        
        if sed is True:
            try:
                self.generate_sed(parms)
                print('SED is generated')
                
            except:
                print('Wrong parameters to generate SED')
                
        if line_spectrum is True:
            try:
                self.generate_line_spectrum(parms)
                print('Line spectrum is generated')
            except:
                print('Wrong parameters to generate line spectra')
        
        if conti is True:
            try:
                self.generate_continuum(parms.conti_parms)
                print('Dust continuum is generated')
            except:
                print('Wrong parameters to generate dust continuum')

    
    def generate_cube(self, parms, cube_parms, **kwargs):
                    
        """
        incl               : Inclination angle of the disk
        line               : Transistion level (see 'molecule_ch3oh.inp')
        v_width            : Range of velocity to simulate
        nlam               : Number of velocities
        npix               : Number of map's pixels
        nodust             : If False, dust effect is included
        scat               : If True and nodust=False, scattering is included. (Time-consuming)
        extracted_gas      : If True, spectral line is extracted (I_{dust+gas}-I_{dust})
        sizeau             : Map's span
        """
        
        condition = {}
        for c in ['nodust', 'scat', 'extract_gas']:
            if c in kwargs:
                condition[c] = kwargs[c]
            else:
                condition[c] = getattr(parms.condition_parms, c, False)
        nodust = condition['nodust']
        scat = condition['scat']
        extract_gas = condition['extract_gas']
        
        npix        = getattr(cube_parms,        'npix',   100)
        sizeau      = getattr(cube_parms,      'sizeau',   100)
        incl        = getattr(cube_parms,        'incl',    70)
        line        = getattr(cube_parms,        'line',   240)
        v_width     = getattr(cube_parms,     'v_width',    10)
        nlam        = getattr(cube_parms,        'nlam',    10)
        vkms        = getattr(cube_parms,        'vkms',     0)
        
        prompt = f'npix {npix} sizeau {sizeau} incl {incl} iline {line} vkms {vkms} widthkms {v_width} linenlam {nlam}'
        
        if nlam > 15:
            type_note = 'pv'
        else:
            type_note = 'channel'
        
        
        if extract_gas is False:
            if nodust is True:
                prompt = prompt + ' noscat nodust'
                f      = '_nodust'
            elif nodust is False:
                if scat is True:
                    f      = '_scat'
                elif scat is False:
                    prompt = prompt +' noscat'
                    f      = '_noscat'
                else:
                    pass
            else:
                pass
                    
            os.system(f"radmc3d image "+prompt)
            
            if (cube_parms.read_cube is True) or (kwargs['read_cube'] is True):
                self.cube = readImage('image.out')
                if self.save_npz is True:
                    if 'fname' in kwargs.keys():
                        self.save_npzfile(self.cube, cube_parms, fname=kwargs['fname'], f=f, note=type_note)
                    else:
                        self.save_npzfile(self.cube, cube_parms, f=f, note=type_note)
            else:
                pass
            
            if self.save_out is True:
                if 'fname' in kwargs.keys():
                    self.save_outfile(cube_parms, fname=kwargs['fname'], f=f, note=type_note)
                else:
                    self.save_outfile(cube_parms, f=f, note=type_note)
            else:
                pass
                    
        elif extract_gas is True:
            os.system(f"radmc3d image "+prompt)
            if cube_parms.read_cube is True:
                self.cube = readImage('image.out')
                if self.save_npz is True:
                    if 'fname' in kwargs.keys():
                        self.save_npzfile(self.cube, cube_parms, fname=kwargs['fname'], f='_scat', note=type_note)
                    else:
                        self.save_npzfile(self.cube, cube_parms, f='_scat', note=type_note)
                else:
                    pass
                
                if self.save_out is True:
                    if 'fname' in kwargs.keys():
                        self.save_outfile(cube_parms, fname=kwargs['fname'], f='_scat', note=type_note)
                    else:
                        self.save_outfile(cube_parms, f='_scat', note=type_note)
                else:
                    pass
            else:
                pass
            
            os.system(f"radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambdarange {self.cube.wav[0]} {self.cube.wav[-1]} nlam {nlam} noline")
            
            if cube_parms.read_cube is True:
                self.conti = readImage('image.out')
                if self.save_npz is True:
                    if 'fname' in kwargs.keys():
                        self.save_npzfile(self.conti, cube_parms, fname=kwargs['fname'], f='_conti', note=type_note)
                    else:
                        self.save_npzfile(self.conti, cube_parms, f='_conti', note=type_note)
                else:
                    pass
                
                if self.save_out is True:
                    if 'fname' in kwargs.keys():
                        self.save_outfile(cube_parms, fname=kwargs['fname'], f='_conti', note=type_note)
                    else:
                        self.save_outfile(cube_parms, f='_conti', note=type_note)
                else:
                    pass
            else:
                pass
            
            if cube_parms.read_cube is True:
                
                self.cube_list = [self.cube, self.conti]
                if self.save_npz is True:
                    if 'fname' in kwargs.keys():
                        self.save_npzfile(self.cube_list, cube_parms, fname=kwargs['fname'], f='_extracted', note=type_note)
                    else:
                        self.save_npzfile(self.cube_list, cube_parms, f='_extracted', note=type_note)
                else:
                    pass
            else:
                pass
        else:
            pass
            
    
    def generate_continuum(self, conti_parms, **kwargs):
        
        type_note = 'conti'
        
        incl   = getattr(conti_parms,   'incl',   70)
        wav    = getattr(conti_parms,    'wav', 1300)
        npix   = getattr(conti_parms,   'npix',  200)
        sizeau = getattr(conti_parms, 'sizeau',  100)
        
        prompt = f'radmc3d image npix {npix} sizeau {sizeau} incl {incl} lambda {wav} noline'
        
        f = '_scat'
        
        if conti_parms.scat is False:
            prompt = prompt + ' noscat'
            f = '_noscat'
        os.system(prompt)
        
        
        if (conti_parms.read_conti is True) or (kwargs['read_conti'] is True):
            self.conti = readImage('image.out')
            if self.save_npz is True:
                if 'fname' in kwargs.keys():
                    self.save_npzfile(self.conti, conti_parms, fname=kwargs['fname'], f=f, note=type_note)
                else:
                    self.save_npzfile(self.conti, conti_parms, f=f, note=type_note)
        else:
            pass
        
        if self.save_out is True:
            if 'fname' in kwargs.keys():
                self.save_outfile(conti_parms, fname=kwargs['fname'], f=f, note=type_note)
            else:
                self.save_outfile(conti_parms, f=f, note=type_note)
        else:
            pass
        
        return
    
    
    
    def generate_sed(self, parms, **kwargs):
        
        return
    
    def generate_line_spectrum(self, parms, **kwargs):
        
        return
    
    
    
    def save_outfile(self, parms, **kwargs):
        """
        This will save whole information of simulation.
        The file type will be '*.out', which the storage may be MB-scale
        """

        dir = getattr(parms, 'dir', './test/')
        
        if 'fname' in kwargs.keys():
            fname = kwargs['fname']
        else:
            fname = getattr(parms, 'fname', 'test')
        
        if 'f' in kwargs.keys():
            f = kwargs['f']
        else:
            f = ''
            
        if 'note' in kwargs.keys():
            head = kwargs['note'] + '_'
        else:
            head = ''
        
        os.makedirs(dir+'outfile/', exist_ok=True)
        os.system('mv image.out '+dir+'outfile/'+head+fname+f+'.out')

    def save_npzfile(self, cube, parms, **kwargs):
        """
        This will only save image cube regardless of other information
        The file tyep will be '*.npz', which the storage may be kB-scale
        """
        
        dir   = getattr(parms, 'dir', './test/')
        
        if 'fname' in kwargs.keys():
            fname = kwargs['fname']
        else:
            fname = getattr(parms, 'fname', 'test')
        
        
        if 'f' in kwargs.keys():
            f = kwargs['f']
        else:
            f = ''
        
        if 'note' in kwargs.keys():
            head = kwargs['note'] + '_'
        else:
            head = ''
        
        os.makedirs(dir+'npzfile/', exist_ok=True)
        
    
        if isinstance(cube, list):
            np.savez(dir+'npzfile/'+head+fname+f+'.npz',
                    imageJyppix = cube[0].imageJyppix - cube[1].imageJyppix,
                    x           = cube[0].x,
                    nx          = cube[0].nx,
                    y           = cube[0].y,
                    ny          = cube[0].ny,
                    wav         = cube[0].wav,
                    freq        = cube[0].freq)
        else:
            np.savez(dir+'npzfile/'+head+fname+f+'.npz',
                    imageJyppix = cube.imageJyppix,
                    x           = cube.x,
                    nx          = cube.nx,
                    y           = cube.y,
                    ny          = cube.ny,
                    wav         = cube.wav,
                    freq        = cube.freq)
        
