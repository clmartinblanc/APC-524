from scipy import *
import numpy as np
import scipy
import gc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import pandas as pd
import numpy as np
import math as m
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.interpolate import griddata, interp1d
from scipy.signal import savgol_filter

import sys, os
from tqdm import tqdm

"""Use customized plotting theme"""
import matplotlib as mpl


class CoordinateConverter:
    @staticmethod
    def pol2cart(rho, phi):
        """
        This function takes in polar coordinates (rho, phi) and converts them
        to Cartesian coordinates (x, y).

        Parameters:
        rho : float or array-like
            The radial distance in polar coordinates.
        phi : float or array-like
            The angle in radians in polar coordinates.

        Returns:
        x : float or array-like
            The x-coordinate in Cartesian coordinates.
        y : float or array-like
            The y-coordinate in Cartesian coordinates.
        """
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    @staticmethod
    def cart_to_wf(p_2d, eta_1d, N, L0, k_, eta_m0):
        """
        p is 2d (in the x, z), eta is 1d (only x)
        """
        eta_1d = eta_1d
        # the mean has been already subtracted
        p_2d_interp = np.zeros([N, N])
        zplot = np.zeros(
            [N, N]
        )  # To show in the original cartesian grid z where the interpolating grid z' is
        for i in range(N):  # For each x
            #
            z = np.linspace(-eta_m0, L0 - eta_m0, N, endpoint=False)
            f = interp1d(z, p_2d[i, :], kind="quadratic", fill_value="extrapolate")
            zeta = z
            zplot[i] = zeta + eta_1d[i] * np.exp(-1 * np.abs(zeta))
            p_2d_interp[i] = f(zplot[i])
            #
            p_1d_interp = np.average(p_2d_interp, axis=-1)
            # average along the x direction
        #
        return p_2d_interp, p_1d_interp, zplot, zeta


'''
def pol2cart(rho, phi):
    """
    This function takes in polar coordinates (rho, phi) and converts them
    to Cartesian coordinates (x, y).

    Parameters:
    rho : float or array-like
        The radial distance in polar coordinates.
    phi : float or array-like
        The angle in radians in polar coordinates.

    Returns:
    x : float or array-like
        The x-coordinate in Cartesian coordinates.
    y : float or array-like
        The y-coordinate in Cartesian coordinates.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
'''


class SpectrumAnalyzer:
    def __init__(self):
        self.variance = []
        self.integral = []
        self.polar_integral = []

    def spectrum_integration_2d(self, eta, N, L, CHECK=False):
        """
        Perform azimuthal integration of a 2D spectrum.

        This function computes the azimuthal integration of a 2D spectrum. It's useful
        for analyzing data in spectral space. The function can also perform a check
        to ensure that the units are consistent and the variance of the data is correctly
        recovered.

        Parameters:
        eta : ndarray
            The 2D data array (e.g., surface elevation).
        N : int
            The number of points in each dimension.
        L : float
            The physical size of the domain.
        CHECK : bool, optional
            If True, prints out intermediate results for verification.

        Returns:
        Various components of the computed spectrum, including wavenumber, the shifted FFT spectrum, the azimuthally integrated spectrum, etc.
        """
        # Implementation of your 2D spectrum_integration function
        # Variance of the input data
        varr = np.var(eta)
        if CHECK:
            print("var", varr)
        variance.append(varr)
        print("mean", np.mean(eta))

        # Definitions for spectral analysis
        T0 = 2 * np.pi
        deltaf = 1 / (2 * np.pi)
        wavenumber = 2 * np.pi * np.fft.fftfreq(N, L / N)
        kx = np.fft.fftshift(wavenumber)
        ky = kx
        kx_tile, ky_tile = np.meshgrid(kx, ky)
        theta = 2 * np.pi * np.arange(-N / 2, N / 2) / N

        # Wavenumber array and differential elements
        k = wavenumber[0 : int(N / 2)]
        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        dk = k[1] - k[0]
        dtheta = theta[1] - theta[0]

        # Fourier transform and normalization
        spectrum = np.fft.fft2(eta) / (N * N) ** 0.5
        F = np.absolute(spectrum) ** 2 / N**2 / (dkx * dky)

        if CHECK:
            print("sum F", np.sum(F))
        F_center = np.fft.fftshift(F, axes=(0, 1))

        # Convert to polar coordinates
        k_tile, theta_tile = np.meshgrid(k, theta)
        kxp_tile, kyp_tile = CoordinateConverter.pol2cart(k_tile, theta_tile)

        # Integration in Cartesian coordinates
        integ = np.sum(F_center) * dkx * dky
        print("integral", integ)
        self.integral.append(integ)

        # Interpolation and azimuthal integration in polar coordinates
        F_center_polar = scipy.interpolate.griddata(
            (kx_tile.ravel(), ky_tile.ravel()),
            F_center.ravel(),
            (kxp_tile, kyp_tile),
            method="nearest",
            fill_value=0,
        )
        F_center_polar_integrated = np.sum(F_center_polar * k_tile, axis=0) * dtheta

        int_pol = np.sum(F_center_polar_integrated) * dk
        if CHECK:
            print("sum polar integrated", int_pol)
        self.polar_integral.append(int_pol)

        # Return computed values
        return (
            k,
            F_center,
            F_center_polar_integrated,
            F_center_polar,
            k_tile,
            kxp_tile,
            kyp_tile,
            theta_tile,
            theta,
            variance,
            integral,
            polar_integral,
            kx,
            ky,
        )

    def spectrum_integration_3d(self, data, L0, N, T):
        Nt = data.shape[0]
        dx = L0 / N  # spatial sampling step along X in (m)
        dy = L0 / N  # spatial sampling step along Y in (m)

        t_max = dt * data.shape[0]  # s
        x_max = dx * data.shape[1]  # m
        y_max = dy * data.shape[2]  # m

        rmax = (x_max**2 + y_max**2) ** 0.5
        nr = Nt  # number intervals in r

        radii = np.linspace(0, rmax, nr)

        x = np.linspace(0, x_max, data.shape[1])  # m
        y = np.linspace(0, y_max, data.shape[2])  # m
        xx, yy = np.meshgrid(x, y, indexing="ij")

        # omega = np.linspace(-np.pi/dt , np.pi/dt , data.shape[0])                          # frequency (Hz)
        theta = np.linspace(0, 2 * np.pi, Nt)

        wavenumber = 2 * np.pi * np.fft.fftfreq(N, L0 / N)
        omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nt, T / Nt))
        kx = np.fft.fftshift(wavenumber)
        ky = kx

        k = wavenumber[0 : int(N / 2)]  # only k>0
        # omega = omega[int(Nt/2):] #only freq>0

        dkx = kx[1] - kx[0]
        dky = ky[1] - ky[0]
        dk = k[1] - k[0]
        dtheta = theta[1] - theta[0]
        domega = omega[1] - omega[0]

        kx_tile, ky_tile = np.meshgrid(kx, ky)  # kx-ky space
        Omega, K = np.meshgrid(omega, k)  # k-omega space

        k_tile, theta_tile = np.meshgrid(k, theta)  # k-theta space
        kxp_tile, kyp_tile = CoordinateConverter.pol2cart(k_tile, theta_tile)

        F_xyomega = np.zeros((N, F_3D.shape[0], N))
        F_kthetaomega = np.zeros((theta.shape[0], F_3D.shape[0], k.shape[0]))
        F_komega = np.zeros((256, F_3D.shape[0]))  # dimension nr x omega

        for i in range(0, F_3D.shape[0]):  # loop in the frequencies(omega)
            F_xy = F_3D[i]  # spectrum F(kx,ky) for each freq i
            F_xyomega[:, i] = F_xy  # each colum of F_xyomega is a freq

            F_ktheta = scipy.interpolate.griddata(
                (kx_tile.ravel(), ky_tile.ravel()),
                F_xy.ravel(),
                (kxp_tile, kyp_tile),
                method="nearest",
            )  # F(omega,kx,ky) --> F(omega,k,theta) for each freq i
            F_kthetaomega[:, i] = F_ktheta  # each colum of F_kthetaomega is a freq

            dtheta = theta[1] - theta[0]
            F_komega[:, i] = (
                np.sum(F_ktheta * k_tile, axis=0) * dtheta
            )  #  integral in theta for each freq i so : F(k,theta) ---> F(k) for each omega ---> acces to F(k,omega)

        return F_xy, F_xyomega, F_ktheta, F_kthetaomega, F_komega

    def spectrum_integration(self, data, N, L, dimension="2D", CHECK=False):
        if dimension == "2D":
            # Análisis 2D
            return self.spectrum_integration_2d(data, N, L, CHECK)
        elif dimension == "3D":
            # Análisis 3D
            return self.spectrum_integration_3d(data, L0, N, T)
        else:
            raise ValueError("Dimension must be '2D' or '3D'")


"""
# example
analyzer = SpectrumAnalyzer()
results_2d = analyzer.spectrum_integration(eta, N, L, dimension='2D', CHECK=True)

analyzer = SpectrumAnalyzer()
results_3d = analyzer.spectrum_integration(data, N, L0, dimension='3D')
"""


'''
variance =[]
integral = []
polar_integral = []

def spectrum_integration(eta, N, L, CHECK=False):
    """
    Perform azimuthal integration of a 2D spectrum.

    This function computes the azimuthal integration of a 2D spectrum. It's useful
    for analyzing data in spectral space. The function can also perform a check
    to ensure that the units are consistent and the variance of the data is correctly
    recovered.

    Parameters:
    eta : ndarray
        The 2D data array (e.g., surface elevation).
    N : int
        The number of points in each dimension.
    L : float
        The physical size of the domain.
    CHECK : bool, optional
        If True, prints out intermediate results for verification.

    Returns:
    Various components of the computed spectrum, including wavenumber, the shifted FFT spectrum, the azimuthally integrated spectrum, etc.
    """
    # Variance of the input data
    varr = np.var(eta)
    if CHECK: print('var', varr)
    variance.append(varr)
    print('mean', np.mean(eta))

    # Definitions for spectral analysis
    T0 = 2*np.pi
    deltaf = 1 / (2 * np.pi)
    wavenumber = 2 * np.pi * np.fft.fftfreq(N, L / N)
    kx = np.fft.fftshift(wavenumber); ky = kx
    kx_tile, ky_tile = np.meshgrid(kx, ky)
    theta = 2 * np.pi * np.arange(-N / 2, N / 2) / N

    # Wavenumber array and differential elements
    k = wavenumber[0:int(N / 2)]
    dkx = kx[1] - kx[0]; dky = ky[1] - ky[0]
    dk = k[1] - k[0]; dtheta = theta[1] - theta[0]

    # Fourier transform and normalization
    spectrum = np.fft.fft2(eta) / (N * N)**0.5
    F = np.absolute(spectrum)**2 / N**2 / (dkx * dky)

    if CHECK: print('sum F', np.sum(F))
    F_center = np.fft.fftshift(F, axes=(0, 1))

    # Convert to polar coordinates
    k_tile, theta_tile = np.meshgrid(k, theta)
    kxp_tile, kyp_tile = pol2cart(k_tile, theta_tile)

    # Integration in Cartesian coordinates
    integ = np.sum(F_center) * dkx * dky
    print('integral', integ)
    integral.append(integ)

    # Interpolation and azimuthal integration in polar coordinates
    F_center_polar = scipy.interpolate.griddata(
        (kx_tile.ravel(), ky_tile.ravel()), F_center.ravel(),
        (kxp_tile, kyp_tile), method='nearest', fill_value=0
    )
    F_center_polar_integrated = np.sum(F_center_polar * k_tile, axis=0) * dtheta

    int_pol = np.sum(F_center_polar_integrated) * dk
    if CHECK: print('sum polar integrated', int_pol)
    polar_integral.append(int_pol)

    # Return computed values
    return k, F_center, F_center_polar_integrated, F_center_polar, k_tile, kxp_tile, kyp_tile, theta_tile, theta, variance, integral, polar_integral, kx, ky

'''


def phase_partion(ux, uy, f, s1_exp, s2_exp):
    ux_air = ux * (1.0 - f) ** s1_exp
    ux_water = ux * f ** (1 / s2_exp)
    uy_air = uy * (1.0 - f) ** s1_exp
    uy_water = uy * f ** (1 / s2_exp)
    return ux_air, ux_water, uy_air, uy_water


#
def interp_2d(xdata, zdata, xtile, ztile, fld):
    fld_int = griddata(
        (xdata.ravel(), zdata.ravel()), fld.ravel(), (xtile, ztile), method="nearest"
    )
    return fld_int


#


class FileHandler:
    @staticmethod
    def load_bin(pfile, N, is_2d):
        if is_2d == 1:
            snapshot = np.fromfile(pfile, dtype=np.float64)
            snapshot = snapshot.reshape([N, N])
            snapshot = np.transpose(snapshot)
        else:
            snapshot = np.fromfile(pfile, dtype=np.float64)
            snapshot = snapshot.reshape([N, N, N])
            snapshot = np.transpose(snapshot)
        return snapshot

    @staticmethod
    def return_file(common_path, name, istep, N, is_2d):
        #
        if is_2d == 1:
            pfile = common_path + "field/" + name + "_2d_avg_" + istep + ".bin"
            file = load_bin(pfile, N, is_2d)
        else:
            pfile = common_path + "field/" + name + "_3d_" + istep + ".bin"
            file = load_bin(pfile, N, is_2d)
        #
        return file


"""
def load_bin(pfile,N,is_2d):
    if is_2d == 1:
      snapshot = np.fromfile(pfile, dtype=np.float64);
      snapshot = snapshot.reshape([N,N]);
      snapshot = np.transpose(snapshot);
    else:
      snapshot = np.fromfile(pfile, dtype=np.float64);
      snapshot = snapshot.reshape([N,N,N]);
      snapshot = np.transpose(snapshot);
    return snapshot
#
def return_file(common_path,name,istep,N,is_2d):
    #
    if is_2d == 1:
      pfile = common_path+'field/'+name+'_2d_avg_'+istep+'.bin';
      file  = load_bin(pfile,N,is_2d);
    else:
      pfile = common_path+'field/'+name+'_3d_'+istep+'.bin';
      file  = load_bin(pfile,N,is_2d);
    #
    return file
#
"""
'''
def cart_to_wf(p_2d,eta_1d,N,L0,k_,eta_m0):
    """
    p is 2d (in the x, z), eta is 1d (only x)
    """
    eta_1d      = eta_1d;         # the mean has been already subtracted
    p_2d_interp = np.zeros([N,N])
    zplot       = np.zeros([N,N]) # To show in the original cartesian grid z where the interpolating grid z' is
    for i in range(N): # For each x
      #
      z        = np.linspace(-eta_m0,L0-eta_m0,N,endpoint=False);
      f        = interp1d(z,p_2d[i,:],kind='quadratic',fill_value = "extrapolate");
      zeta     = z;
      zplot[i] = zeta + eta_1d[i]*np.exp(-1*np.abs(zeta));
      p_2d_interp[i] = f(zplot[i]);
      #
    p_1d_interp = np.average(p_2d_interp,axis=-1); # average along the x direction
    #
    return p_2d_interp, p_1d_interp, zplot, zeta
'''


class Utilities:
    @staticmethod
    def mom_flux_p(pre, nx, ny, nz):
        """
        Compute the mean pressure and the flux due to pressure
        """
        pre_mean = np.average(pre)

        mf_px = -np.average((pre[:] - pre_mean) * nx[:])
        mf_py = -np.average((pre[:] - pre_mean) * ny[:])
        mf_pz = -np.average((pre[:] - pre_mean) * nz[:])

        return mf_px, mf_py, mf_pz

    @staticmethod
    def mom_flux_p_alt(pre_1d, eta_1d, L0, N):
        """
        Compute momentum flux due to pressure
        """
        etahat = savgol_filter(eta_1d, 31, 4)
        eps = np.gradient(etahat) / (L0 / N)
        mf_px = np.average(pre_1d[:] * eps[:])
        #
        return mf_px

    @staticmethod
    def mom_flux_v_alt(Sxx_1d, Sxy_1d, eta_1d, L0, N, mu2):
        """
        Compute momentum flux due to dissipation
        """
        etahat = savgol_filter(eta_1d, 31, 4)
        eps = np.gradient(etahat) / (L0 / N)
        mf_vx = mu2 * np.average(Sxy_1d[:] - Sxx_1d[:] * eps[:])
        #
        return mf_vx

    @staticmethod
    def mom_flux_v(Sxx, Sxy, Sxz, Syy, Syz, Szz, nx, ny, nz, mu2):
        """ "
        Compute momentum flux due to viscous dissipation
        """
        mf_vx = +mu2 * np.average(Sxx[:] * nx[:] + Sxy[:] * ny[:] + Sxz[:] * nz[:])
        mf_vy = +mu2 * np.average(Sxy[:] * nx[:] + Syy[:] * ny[:] + Syz[:] * nz[:])
        mf_vz = +mu2 * np.average(Sxz[:] * nx[:] + Syz[:] * ny[:] + Szz[:] * nz[:])
        #
        return mf_vx, mf_vy, mf_vz

    @staticmethod
    def ene_flux_p(pre, u_x, u_y, u_z, nx, ny, nz):
        """
        Compute the mean pressure pre_mean, the mean velocity in each direction and the energy flux due to pressure
        """
        pre_mean = np.average(pre)
        u_x_mean = np.average(u_x)
        u_y_mean = np.average(u_y)
        u_z_mean = np.average(u_z)
        #
        # compute energy flux due to pressure
        #
        en_p = -np.average(
            (pre[:] - pre_mean) * (u_x[:] - u_x_mean) * nx[:]
            + (pre[:] - pre_mean) * (u_y[:] - u_y_mean) * ny[:]
            + (pre[:] - pre_mean) * (u_z[:] - u_z_mean) * nz[:]
        )
        #
        return en_p

    @staticmethod
    def ene_flux_v(Sxx, Sxy, Sxz, Syy, Syz, Szz, u_x, u_y, u_z, nx, ny, nz, mu2):
        """
        Compute energy flux due to viscous dissipation
        """
        u_x_mean = np.average(u_x)
        u_y_mean = np.average(u_y)
        u_z_mean = np.average(u_z)

        en_v = +mu2 * np.average(
            (Sxx[:] * nx[:] + Sxy[:] * ny[:] + Sxz[:] * nz[:]) * (u_x[:] - u_x_mean)
            + (Sxy[:] * nx[:] + Syy[:] * ny[:] + Syz[:] * nz[:]) * (u_y[:] - u_y_mean)
            + (Sxz[:] * nx[:] + Syz[:] * ny[:] + Szz[:] * nz[:]) * (u_z[:] - u_z_mean)
        )
        #
        return en_v

    @staticmethod
    def get_amp(eta, k, L0):
        """
        Function to obtain de stepness ak from the height eta
        """
        ak = k * ((2 / (L0**2)) * np.std((eta[:]) ** 2)) ** 0.5
        return ak


'''
def mom_flux_p(pre,nx,ny,nz):
    """
    Compute the mean pressure and the flux due to pressure
    """
    pre_mean = np.average(pre);

    mf_px = - np.average( (pre[:]-pre_mean)*nx[:] );
    mf_py = - np.average( (pre[:]-pre_mean)*ny[:] );
    mf_pz = - np.average( (pre[:]-pre_mean)*nz[:] );

    return mf_px,mf_py,mf_pz
#
def mom_flux_p_alt(pre_1d,eta_1d,L0,N):
    """
    Compute momentum flux due to pressure
    """
    etahat = savgol_filter(eta_1d, 31, 4);
    eps    = np.gradient(etahat)/(L0/N);
    mf_px  = np.average( pre_1d[:]*eps[:] );
    #
    return mf_px
#
def mom_flux_v_alt(Sxx_1d,Sxy_1d,eta_1d,L0,N,mu2):
    """
    Compute momentum flux due to dissipation
    """
    etahat = savgol_filter(eta_1d, 31, 4);
    eps    = np.gradient(etahat)/(L0/N);
    mf_vx  = mu2*np.average( Sxy_1d[:] - Sxx_1d[:]*eps[:] );
    #
    return mf_vx
#
def mom_flux_v(Sxx,Sxy,Sxz,Syy,Syz,Szz,nx,ny,nz,mu2):
    """"
    Compute momentum flux due to viscous dissipation
    """
    mf_vx = + mu2*np.average( (Sxx[:]*nx[:] + Sxy[:]*ny[:] + Sxz[:]*nz[:]) );
    mf_vy = + mu2*np.average( (Sxy[:]*nx[:] + Syy[:]*ny[:] + Syz[:]*nz[:]) );
    mf_vz = + mu2*np.average( (Sxz[:]*nx[:] + Syz[:]*ny[:] + Szz[:]*nz[:]) );
    #
    return mf_vx,mf_vy,mf_vz
#
def ene_flux_p(pre,u_x,u_y,u_z,nx,ny,nz):
    """"
    Compute the mean pressure pre_mean, the mean velocity in each direction and the energy flux due to pressure
    """"
    pre_mean = np.average(pre);
    u_x_mean = np.average(u_x);
    u_y_mean = np.average(u_y);
    u_z_mean = np.average(u_z);
    #
    # compute energy flux due to pressure
    #
    en_p = - np.average( (pre[:]-pre_mean)*(u_x[:]-u_x_mean)*nx[:] +
                         (pre[:]-pre_mean)*(u_y[:]-u_y_mean)*ny[:] +
                         (pre[:]-pre_mean)*(u_z[:]-u_z_mean)*nz[:] );
    #
    return en_p
#
def ene_flux_v(Sxx,Sxy,Sxz,Syy,Syz,Szz,u_x,u_y,u_z,nx,ny,nz,mu2):
    """
    Compute energy flux due to viscous dissipation
    """
    u_x_mean = np.average(u_x);
    u_y_mean = np.average(u_y);
    u_z_mean = np.average(u_z);

    en_v = + mu2*np.average( ( Sxx[:]*nx[:]+Sxy[:]*ny[:]+Sxz[:]*nz[:] )*(u_x[:]-u_x_mean) +
                             ( Sxy[:]*nx[:]+Syy[:]*ny[:]+Syz[:]*nz[:] )*(u_y[:]-u_y_mean) +
                             ( Sxz[:]*nx[:]+Syz[:]*ny[:]+Szz[:]*nz[:] )*(u_z[:]-u_z_mean) );
    #
    return en_v
#
def get_amp(eta,k,L0):
    """
    Function to obtain de stepness ak from the height eta
    """
    ak = k*( (2/(L0**2))*np.std( (eta[:])**2) )**0.5;
    return ak
'''
