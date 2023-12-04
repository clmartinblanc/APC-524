""" This file defines a few useful classes:
    Case: metadate of cases.
    Interface: interface related info read-in. 
"""

import pandas as pd
import numpy as np
import math
import sys, os
sys.path.append('/projects/DEIKE/jiarongw/jiarongw-postprocessing/jupyter_notebook/functions/')
sys.path.append('/projects/DEIKE/jiarongw/jiarongw-postprocessing/jupyter_notebook/project_specific/windwave/')
# sys.path.append('/home/jiarong/research/postprocessing/jupyter_notebook/functions/')
from tqdm import tqdm
from prepare import load_object, save_object

""" Case """
sys.path.append('/home/jiarong/research/postprocessing/jupyter_notebook/project_specific/windwave/')
from helper import RealWave

class Case():
    """
        Class for each case. Initialization takes in the metadata like Bo, Re, ak etc.
        self.availt: The longest running time
        self.path: Path of the case file
    """
    def __init__(self, ustar, Retau, Bo, g, ak, LEVEL, emax, alterMU=1., L0=2*np.pi, k=4., h=1., OUTLEVEL=8,
                 working_dir='/home/jiarong/research/projects/turbulence/', prefix='curved_fixREtau_', postfix='/', 
                 PRINTWAVE=False, NOMATCH=True, PRECURSOR=False):
        '''
        alterMU: the altered MU ratio. Default 1. Turn on NOMATCH as well.
        self.N: output grid number
        '''
        # Register the metadata and spell the path
        if PRECURSOR:
            self.ustar = ustar; self.ak = ak; self.Bo = None; self.Retau = Retau; self.alterMU = None
            self.emax=emax; self.LEVEL = LEVEL; self.N = 2**OUTLEVEL
            self.L0 = L0; self.g = None; self.k = k; self.h = h
            self.rho1 = 1; self.rho2 = 1.225/1000.; self.sigma = None
            self.mu2 = self.ustar*self.rho2*(self.L0-h)/self.Retau; self.mu1 = self.mu2/(18.31e-6/10.0e-4)/alterMU
            self.path = working_dir + prefix + 'REtau%g_ak%g_LEVEL%g_emax%g' %(self.Retau,self.ak,self.LEVEL,self.emax) + postfix
        else:   
            self.ustar = ustar; self.ak = ak; self.Bo = Bo; self.Retau = Retau; self.alterMU = alterMU
            self.emax=emax; self.LEVEL = LEVEL; self.N = 2**OUTLEVEL
            self.L0 = L0; self.g = g; self.k = k; self.h = h
            self.rho1 = 1; self.rho2 = 1.225/1000.; self.sigma = self.g/(self.Bo*self.k**2)
            self.mu2 = self.ustar*self.rho2*(self.L0-h)/self.Retau; self.mu1 = self.mu2/(18.31e-6/10.0e-4)/alterMU
            if NOMATCH == True:
                self.path = working_dir + prefix + 'REtau%g_BO%g_g%g_ak%g_MU%g_LEVEL%g_emax%g' %(self.Retau,self.Bo,self.g,self.ak,self.alterMU,self.LEVEL,self.emax) + postfix
            else:
                self.path = working_dir + prefix + 'REtau%g_BO%g_g%g_ak%g_LEVEL%g_emax%g' %(self.Retau,self.Bo,self.g,self.ak,self.LEVEL,self.emax) + postfix
            # Run wave helper function to compute wave related info0
            # Notice that this depends on the definition of the wave in the specific set of cases
            self.wave = RealWave(g=self.g, sigma=self.sigma, rho=self.rho1, rho_air=self.rho2, mu = self.mu1, mu_air = self.mu2)
            self.wave.k2omega(self.k)       
            # Print out wave info; to double check, print the message file
            if PRINTWAVE:
                print(self.path)
                print("mu1 = %g, rho1 = %g, mu2 = %g, rho2 = %g, sigma = %g" %(self.mu1, self.rho1, self.mu2, self.rho2, self.sigma))
                print("Given k = %g (1/m), calculated omega = %g (1/s), period = %g (s), phase speed c = %g (m/s), wavelength = %g (m), Bo = %g" 
                      %(self.wave.k, self.wave.omega, 2*np.pi/self.wave.omega, self.wave.c, self.wave.wl, self.Bo))
                f = open(self.path+"message", "r")
                print(f.read())
    
    def eta_series(self, nframe, tstart, dt=1, PRUNING=True):
        '''
        This function reads in a eta time series and create a Eta object for each time.
        Input:
            nframe: Number of total frames.
            tstart: The starting time.
            dt: Time interval between each read-in.
            PRUNING: If eta is output by multiple processes and have multiple headers
            (might become obsolete).
        Output:
            self.energy_t: energy (std(eta)**2) time series (scalar)
            self.interface_t: time series of Interface object       
        '''       
        self.t = np.zeros(nframe)
        self.energy_t = []
        self.interface_t = []
        for i in tqdm (range (0,nframe)):
            self.t[i] = tstart+i*dt
            interface = Interface(self.L0, self.N, self.path, self.t[i], PRUNING=PRUNING)
            self.interface_t.append(interface)
            self.energy_t.append(np.std(interface.eta_tile)**2)      
        
        self.energy_t = np.array(self.energy_t)
    
    def mean_profile(self, time):
        NSLICE = 256
        NGRID = self.N
        self.yarray = np.linspace(0,self.L0,self.N,endpoint=False)+self.L0/2**self.N/2
        self.ux_ensemble = []
        for t in time:
            ux_3D = [] # axis0 in z, axis1 in x, axis2 in y  (in the code)
            for i in range (0,NSLICE-1):
                filename = self.path + 'field/ux_t%g_slice%g' % (t,i)
                snapshot = np.loadtxt(filename, dtype = np.str, delimiter='\t')
                snapshot.reshape([NGRID,NGRID+1])
                ux = snapshot[:,0:NGRID].astype(np.float)
                ux_3D.append(ux)
            ux_3D = np.array(ux_3D)
            ux_aver = np.zeros(NGRID)
            # Slice in x,z(y) plane and average
            for i in range(0,NGRID):
                ux_aver[i] = np.average(ux_3D[:,:,i])
            self.ux_ensemble.append(ux_aver)
        self.ux_ensemble_aver = np.average(np.array(self.ux_ensemble), axis = 0)
        
    def field(time): 
        pass
    
""" Interface2D """
from scipy.interpolate import griddata
import gc
from scipy.special import gamma
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
from scipy.interpolate import griddata

class Interface():
    '''
    Class for every interface related output. Unstructured grid.   
    '''  
    def __init__(self, L0, N, path, t, PRUNING=True):
        '''
        Input:
            L0, N: The desired output grid number
            working_dir: The case's directory
            t: Time of this eta file.
            PRUNING: If eta is output by multiple processes and have multiple headers
                    (might become obsolete).    
        Output:
            self.eta_tile: 2D interpolated eta
            self.p: 2D interpolated surface pressure
            self.tau: 2D interpolated surface viscous stress
        '''
        self.L0 = L0; self.N = N; self.t = t
        xarray = np.linspace(-self.L0/2.,self.L0/2.,self.N,endpoint=False)+self.L0/2**self.N/2 #Centered grid
        zarray = np.linspace(-self.L0/2.,self.L0/2.,self.N,endpoint=False)+self.L0/2**self.N/2 #size of self.N*self.N
        x_tile, z_tile = np.meshgrid(xarray,zarray)
        filename = path + 'eta/eta_loc_t%g' %self.t
        snapshot = pd.read_table(filename, delimiter = ',')
        if PRUNING:
            snapshot = snapshot[snapshot.x != 'x']
            snapshot = snapshot.astype('float')
            snapshot = snapshot[snapshot.pos < 1 + 0.4/4] # Exclude data over slope 0.4
        # Interpolate over x-z plane
        xdata = np.array(snapshot.x, dtype=float); zdata = np.array(snapshot.z, dtype=float); 
        etadata = np.array(snapshot.pos, dtype=float); 
        del (snapshot); gc.collect()        
        self.eta_tile = griddata((xdata.ravel(), zdata.ravel()), etadata.ravel(), (x_tile, z_tile), method='nearest')

class Interface2D():
    """Class for every interface related 2D output. Unstructured grid input.
            
    Attributes:
        xarray: equal distanced x grid 
        zarray: equal distanced y grid 
        <field>data: row data of <field>
        <field>: interpolated data of <field>, including eta/p/grad/dudy/uxw...    
    """
     
    def __init__(self, L0, N, path, t, PRUNING=True, pruningz=1+0.4/4., pre='field/eta_loc_t', filename=None):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
            Do not include the `self` parameter in the ``Args`` section.

        Args:
            L0, N: The desired output grid number
            working_dir: The case's directory
            t: Time of this eta file.
            PRUNING: If eta is output by multiple processes and have multiple headers
                    (only applicable to MPI processed file).  
            pruningz: the height above which the points are discarded
            pre: the prefix of the desirable data file.
            filename: directly give filename instead of creat based on time
        """
        self.L0 = L0; self.N = N; self.t = t
        self.xarray = np.linspace(-self.L0/2.,self.L0/2.,self.N,endpoint=False)+self.L0/2**self.N/2 # Centered grid for interpolation
        self.zarray = np.linspace(-self.L0/2.,self.L0/2.,self.N,endpoint=False)+self.L0/2**self.N/2 # Centered grid for interpolation
        if filename == None:
            filename = path + pre + '%g' %self.t
            snapshot = pd.read_table(filename, delimiter = ',')
        else:           
            snapshot = pd.read_table(filename, delimiter = ',')
        snapshot = pd.read_table(filename, delimiter = ',')
        # Field entries
        # x,pos,epsilon,p,p_p1,p_p2,dudy1,dudy2,dvdx1,dvdx2,dudx1,dudx2,dvdy1,dvdy2,uxa,uya,uxw,uyw
        # Updated: x, pos, epilon,p,dudy,dvdx,dudx,dvdy,uxa,uya,uxw,uyw
        if PRUNING:
            snapshot = snapshot[snapshot.x != 'x']
            snapshot = snapshot.astype('float')
            print('Pruning points above %g!' %pruningz)
            snapshot = snapshot[snapshot.pos < pruningz] # Exclude data over slope 0.4
            snapshot = snapshot[abs(snapshot.p-snapshot.p.mean()) < 10**(-1)] # Extra pruning for wild p
            snapshot = snapshot[np.isinf(snapshot.epsilon) == 0] # Gradient showing inf
            
        snapshot = snapshot.sort_values(by = ['x'])      
        
        self.xdata = np.array(snapshot.x, dtype=float)
        self.zdata = np.array(snapshot.z, dtype=float)
        self.etadata = np.array(snapshot.pos, dtype=float)
        self.pdata = np.array(snapshot.p, dtype=float)
        self.graddata = np.array(snapshot.epsilon, dtype=float)
        self.dudydata = np.array(snapshot.dudy, dtype=float)
        self.dvdxdata = np.array(snapshot.dvdx, dtype=float)
        self.dudxdata = np.array(snapshot.dudx, dtype=float)
        self.dvdydata = np.array(snapshot.dvdy, dtype=float)
#         self.uxwdata = np.array(snapshot.uxw, dtype=float)
#         self.uywdata = np.array(snapshot.uyw, dtype=float)
#         self.delta = np.array(snapshot.delta, dtype=float)

        del (snapshot); gc.collect()  # Only necessary for 2D for memory issue              
        
        # Interpolate over x and z, 'nearest' is used to ensure that none of the interpolated point is 'nan'
        self.xtile, self.ztile = np.meshgrid(self.xarray,self.zarray)
        self.eta = griddata((self.xdata.ravel(), self.zdata.ravel()), self.etadata.ravel(), (self.xtile, self.ztile), method='nearest')
        self.p = griddata((self.xdata.ravel(), self.zdata.ravel()), self.pdata.ravel(), (self.xtile, self.ztile), method='nearest')
        self.dudy = griddata((self.xdata.ravel(), self.zdata.ravel()), self.dudydata.ravel(), (self.xtile, self.ztile), method='nearest')
        self.dvdx = griddata((self.xdata.ravel(), self.zdata.ravel()), self.dvdxdata.ravel(), (self.xtile, self.ztile), method='nearest')
        self.dudx = griddata((self.xdata.ravel(), self.zdata.ravel()), self.dudxdata.ravel(), (self.xtile, self.ztile), method='nearest')
        self.dvdy = griddata((self.xdata.ravel(), self.zdata.ravel()), self.dvdydata.ravel(), (self.xtile, self.ztile), method='nearest')
        self.grad = griddata((self.xdata.ravel(), self.zdata.ravel()), self.graddata.ravel(), (self.xtile, self.ztile), method='nearest')
#         self.uxw = griddata(self.xdata.ravel(), self.zdata.ravel(), self.uxwdata.ravel(), self.xarray, self.zarray, method='nearest')
#         self.uyw = griddata(self.xdata.ravel(), self.zdata.ravel(), self.uywdata.ravel(), self.xarray, self.zarray, method='nearest')
        del(self.etadata); del(self.pdata); del(self.dudydata); del(self.dvdxdata); del(self.dudxdata)
        del(self.dvdydata); del(self.graddata)
        
        # Get the phase index
        # axis 0 is y, axis 1 is x
        # TODO: MAKE THIS 2D COMPATIBLE
        eta_1D = np.average(self.eta, axis=0)
        eta_1D_filtered = eta_1D-np.average(eta_1D)
        # CAUTION: It seems like doing filtering will make the phase inaccurate
#         eta_1D_filtered = self.__butter_lowpass_filter(eta_1D-np.average(eta_1D))
        analytic_signal = hilbert(eta_1D_filtered)
        self.phase = np.angle(analytic_signal)
        self.idx = abs(self.phase).argmin() # 0 corresponds to crest, -pi/pi corresponds to trough
#         from scipy.signal import argrelextrema
#         maxm1 = argrelextrema(self.phase, np.greater)
#         self.idx = maxm1[0][3]
    
    def __butter_lowpass_filter(self, data, CUT=4):
        """A helper function that performs lowpass filtering."""
        T = 1           # Sample Period
        fs = self.N        # Sample rate, Hz (should be the xarray size)
        cutoff = CUT    # desired cutoff frequency of the filter, Hz
        nyq = 0.5 * fs  # Nyquist Frequency
        order = 4       # sin wave can be approximately represented as quadratic
        n = int(T * fs) # total number of samples
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    def uwater(self, c, ustar, omega, Re):
        """ Water velocity decomposition.
            Dependency: gamma from scipy.special, hilbert from scipy.signal
            
            Args:
                c: wave speed (analyical from case.wave.c)
                ustar: u*/c (from case.ustar)
                omega: omega (from case.wave.omega)
                Re: Re (from case.Re)
                
            Attributes:
                uxw_smooth: smoothed direct output of uxw
                uyw_smooth: smoothed direct output of uyw
                eta_smooth
                phase
                ux_orbit: orbital velocity from eta_smooth, u component (simplest estimation)
                uy_orbit: orbital velocity from eta_smooth, v component
                ud: uxw_smooth - ux_orbit, approximate drift velocity (phase dependent)
                ud_analy: analytical time dependent drift, constant along x
                uxw_analy: ud_analy + ux_orbit
        """
        
        # Smooth out simulation output water velocity
        self.uxw_smooth = self.__butter_lowpass_filter(self.uxw, CUT=8)
        self.uyw_smooth = self.__butter_lowpass_filter(self.uyw, CUT=8)

        # The analytical water velocity
        self.eta_smooth = self.__butter_lowpass_filter(self.eta, CUT=8)
        analytic_signal = hilbert(self.eta_smooth)
        self.phase = np.unwrap(np.angle(analytic_signal))
        self.ux_orbit = self.eta_smooth*c*2*np.pi
        self.uy_orbit = self.eta_smooth*c*2*np.pi/np.cos(self.phase)*np.cos(self.phase-np.pi/2)        
        self.ud = self.uxw_smooth - self.ux_orbit
        self.ud_analy = self.t**0.5*gamma(1)/gamma(3/2)/850 * (ustar*c)**2 * (2*np.pi/omega/(1/Re))**0.5 # Drift according to theoretical solution
        self.uxw_analy = self.ud_analy + self.ux_orbit
        
    def stress(self, tau0, mu_a):
        """ Integrate 2D pressure p times surface gradient to get the form drag; strain dudy, dvdx to get the shear stress.
            No quasi 1D approximation is needed.
            Smoothing is optional.
        
            Args:
                tau0: the set total stress, for normalization.
                mu_a: the air dynamic viscosity, for calculating shear stress.
                
            Attributes:
                tau0: rho u_*^2 from given
                p, p_smooth: the original p subtracted by average and smoothed p              
                tau_nux, tau_nuy:
                tau_nux_smooth, tau_nuy_smooth:
        """
        # Pressure
        self.tau0 = tau0
        self.p = self.p-np.average(self.p) # Subtract mean
        # self.p_smooth = self.__butter_lowpass_filter(self.p, CUT=8) # The cut frequency is pretty high to capture rapid change
        self.formdrag = np.average(self.p*self.grad)
        
        # Shear stress
        # Take also dvdx into account
        # tau_nux1 = self.__butter_lowpass_filter(self.dudy_tile, CUT=4)*mu_a
        self.tau_nux1 = self.dudy*mu_a; self.tau_nux2 = self.dvdx*mu_a
        self.tau_nux = (self.dudy+self.dvdx)*mu_a 
#         self.tau_nux_smooth = self.__butter_lowpass_filter(self.tau_nux, CUT=8)
        self.tau_nuy = 2*self.dvdy*mu_a
#         self.tau_nuy_smooth = self.__butter_lowpass_filter(self.tau_nuy, CUT=8)
        self.shear = np.average(self.tau_nux)
    
    def spectrum(self,peak=4):
        """2D Fourier transform to get the peak frequency energy. Can later be extended to include other spectrum analysis.
        
            Args:
                peak: int. Index of the peak frequency in the spectrum array.
                
            Attributes:
                Ep: peak frequency energy.
        """
        spectrum = np.fft.fft2(self.eta-np.average(self.eta))*(1/self.N)**2 # TODO: How to normalize here?!
        F = np.absolute(spectrum)
        self.Ep = np.average(F,axis=0)[peak] # Only record the peak frequency spetrum amplitude
        
#     def vis(self):
#         """Visualize velocity and stress. Can only be run after uwater and stress are both run."""
#         plt.figure(figsize=[4,6])
        
#         # Plot water velocity in axis1
#         ax1 = plt.subplot(311)       
#         ax1.plot(self.xdata, self.uxwdata, c='C0', alpha = 0.5) # Water velocity uxw unfiltered 
#         ax1.plot(self.xdata, self.uywdata, c='C1', alpha = 0.5) # Water velocity uyw unfiltered
#         ax1.plot(self.xarray, self.uxw_smooth, c='C0', label = '$u_s$') # Water velocity uxw smoothed 
#         ax1.plot(self.xarray, self.uyw_smooth, c='C1', label = '$v_s$') # Water velocity uyw smoothed         
#         ax1.plot(self.xarray, self.ux_orbit, '--', c='C2', label='$u_{orbit}$', alpha = 0.5) # Orbital velocity u analytical
#         ax1.plot(self.xarray, self.uy_orbit, '--', c='C1', label='$v_{orbit}$', alpha = 0.5) # Orbital velocity v analytical
#         ax1.plot(self.xarray, self.ud, '--', c='C3', label = '$u_s - u_{orbit}$', alpha = 0.5)
#         ax1.plot(self.xarray, self.eta_smooth, c='gray', alpha = 0.5)
#         ax1.set_xlabel(r'$x/\lambda$')
#         ax1.set_ylabel(r'$u_s,v_s$')
#         ax1.set_xlim([-0.5,0.5])
#         ax1.set_ylim([-0.05,0.1])
# #         ax1.legend(bbox_to_anchor=(1.02, 0.5), loc = 'center left')   
#         ax1.legend(loc='upper right')   
        
#         # Plot pressure in axis2
#         ax2 = plt.subplot(312)
#         ax2.plot(self.xarray, self.p/self.tau0,  c='C4', alpha=0.5) # Pressure
#         ax2.plot(self.xarray, self.p_smooth/self.tau0, c='C4', label = r'$p$') # Pressure smoothed
#         # Average tau_p
#         ax2.plot(self.xarray, self.p_smooth*self.grad/self.tau0, c='C0', label = r'$p\frac{\partial \eta}{\partial x}$')
#         ax2.plot(self.xarray, np.average(self.p_smooth*self.grad)/self.tau0*np.ones(self.N), '--', c='C0')
#         ax2.plot(self.xarray, self.eta_smooth*50, c='gray', alpha = 0.5)
#         ax2.set_xlim([-0.5,0.5])
#         ax2.set_ylim([-10,10])
#         ax2.set_xlabel(r'$x/\lambda$'); ax2.set_ylabel(r'$p/\rho_a u_*^2$')  
#         ax2.legend(loc='upper right')
        
#         # Plot shear stress in axis 3
#         ax3 = plt.subplot(313)
#         ax3.plot(self.xarray, self.tau_nux/self.tau0, alpha = 0.5) 
#         ax3.plot(self.xarray, self.tau_nux_smooth/self.tau0, c='C0',
#                  label = r'$\tau_{\nu x} = \mu_a (\frac{\partial u}{\partial y}+\frac{\partial v}{\partial x})$') 
#         # Sanity check
#         # ax3.plot(self.xarray, self.tau_nux1/tau0, '--', c='C0') 
#         # Average tau_nu
#         ax3.plot(self.xarray, np.average(self.tau_nux)/self.tau0*np.ones(self.N), '--', c='C0')         
#         ax3.plot(self.xarray, self.tau_nuy/self.tau0, alpha = 0.5) griddata() got multiple values for argument 'method'
#         ax3.plot(self.xarray, self.tau_nuy_smooth/self.tau0, c='C1',
#                  label = r'$\tau_{\nu y} = 2 \mu_a \frac{\partial v}{\partial y}$') 
#         ax3.plot(self.xarray, self.eta_smooth*10, c='gray', alpha = 0.5)
#         ax3.set_xlim([-0.5,0.5])
#         ax3.set_ylim([-0.5,2])
#         ax3.set_xlabel(r'$x/\lambda$'); ax3.set_ylabel(r'$\tau_\nu/\rho_a u_*^2$')
#         ax3.legend(loc='upper right')
#         return (ax1,ax2,ax3)
        
#     def integrate(self):
#         """Compute the phase average input terms.
        
#             Attributes:
#                 S_taunu: total input from shear stress (from smoothed)
#                 S_taunu_err: total input from shear stress unsmoothed, for error bar
#                 S_taunu_w: approximate sub part of S_taunu from wave, computed using ux_orbit uy_orbit
#                 S_taunu_d: appreximate sub part of S_taunu from drift 
#         """
#         self.S_taunu = np.sum(self.tau_nux_smooth*self.uxw_smooth+self.tau_nuy_smooth*self.uyw_smooth)*self.L0/self.N
#         self.S_taunu_err = np.sum(self.tau_nux*self.uxw+self.tau_nuy*self.uyw)*self.L0/self.N #For error bar
#         self.S_taunu_w = np.sum(self.tau_nux_smooth*self.ux_orbit+self.tau_nuy_smooth*self.uy_orbit)*self.L0/self.N 
#         self.S_taunu_d = np.sum(self.tau_nux_smooth*self.ud)*self.L0/self.N
#         self.S_p = np.sum(-self.p_smooth*self.uyw_smooth)*self.L0/self.N #NOTE: this is only the first order approximation!
#         self.S_p_err = np.sum(-self.p*self.uyw)*self.L0/self.N #Might need to add the metrics and uxw
                                                                         
        
#     def interval_average():
#         # How to do interval average of the above calculation
#         pass

def read_eta(case, time):
    # Read in data and some cleaning
    filename = case.path + 'eta/eta_loc_t%.2f' % time  # Adjust the formatting to include two decimal places
    snapshot = pd.read_table(filename, delimiter=',')
    snapshot = snapshot[snapshot.x != 'x']
    snapshot = snapshot.astype('float')
    snapshot = snapshot[snapshot.pos < 1 + 0.4 / 4]
    snapshot = snapshot.sort_values(by=['x'])

    # Interpolate unstructured grid data onto uniform grid
    from scipy.interpolate import griddata
    L0 = np.pi * 2.
    N = 512
    xarray = np.linspace(-L0 / 2., L0 / 2., N, endpoint=False) + L0 / 2 ** N / 2  # Centered grid
    zarray = np.linspace(-L0 / 2., L0 / 2., N, endpoint=False) + L0 / 2 ** N / 2  # size of self.N*self.N
    x_tile, z_tile = np.meshgrid(xarray, zarray)
    xdata = np.array(snapshot.x)
    zdata = np.array(snapshot.z)
    etadata = np.array(snapshot.pos)
    eta_tile = griddata((xdata.ravel(), zdata.ravel()), etadata.ravel(), (x_tile, z_tile), method='nearest')
    return eta_tile