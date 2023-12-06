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
from scipy.signal import butter,filtfilt , hilbert

import sys, os
from tqdm import tqdm
"""Use customized plotting theme"""
import matplotlib as mpl

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
