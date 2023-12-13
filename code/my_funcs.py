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


#
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


#
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


#
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


#
def get_amp(eta, k, L0):
    """
    Function to obtain de stepness ak from the height eta
    """
    ak = k * ((2 / (L0**2)) * np.std((eta[:]) ** 2)) ** 0.5
    return ak


#
def interp_2d(xdata, zdata, xtile, ztile, fld):
    fld_int = griddata(
        (xdata.ravel(), zdata.ravel()), fld.ravel(), (xtile, ztile), method="nearest"
    )
    return fld_int


#
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


#
def mom_flux_p(pre, nx, ny, nz):
    """
    Compute the mean pressure and the flux due to pressure
    """
    pre_mean = np.average(pre)

    mf_px = -np.average((pre[:] - pre_mean) * nx[:])
    mf_py = -np.average((pre[:] - pre_mean) * ny[:])
    mf_pz = -np.average((pre[:] - pre_mean) * nz[:])

    return mf_px, mf_py, mf_pz


#
def mom_flux_p_alt(pre_1d, eta_1d, L0, N):
    """
    Compute momentum flux due to pressure
    """
    etahat = savgol_filter(eta_1d, 31, 4)
    eps = np.gradient(etahat) / (L0 / N)
    mf_px = np.average(pre_1d[:] * eps[:])
    #
    return mf_px


#
def mom_flux_v_alt(Sxx_1d, Sxy_1d, eta_1d, L0, N, mu2):
    """
    Compute momentum flux due to dissipation
    """
    etahat = savgol_filter(eta_1d, 31, 4)
    eps = np.gradient(etahat) / (L0 / N)
    mf_vx = mu2 * np.average(Sxy_1d[:] - Sxx_1d[:] * eps[:])
    #
    return mf_vx


#
def mom_flux_v(Sxx, Sxy, Sxz, Syy, Syz, Szz, nx, ny, nz, mu2):
    """ "
    Compute momentum flux due to viscous dissipation
    """
    mf_vx = +mu2 * np.average(Sxx[:] * nx[:] + Sxy[:] * ny[:] + Sxz[:] * nz[:])
    mf_vy = +mu2 * np.average(Sxy[:] * nx[:] + Syy[:] * ny[:] + Syz[:] * nz[:])
    mf_vz = +mu2 * np.average(Sxz[:] * nx[:] + Syz[:] * ny[:] + Szz[:] * nz[:])
    #
    return mf_vx, mf_vy, mf_vz


def phase_partion(ux, uy, f, s1_exp, s2_exp):
    ux_air = ux * (1.0 - f) ** s1_exp
    ux_water = ux * f ** (1 / s2_exp)
    uy_air = uy * (1.0 - f) ** s1_exp
    uy_water = uy * f ** (1 / s2_exp)
    return ux_air, ux_water, uy_air, uy_water


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


#
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


variance = []
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
    kxp_tile, kyp_tile = pol2cart(k_tile, theta_tile)

    # Integration in Cartesian coordinates
    integ = np.sum(F_center) * dkx * dky
    print("integral", integ)
    integral.append(integ)

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
    polar_integral.append(int_pol)

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




# for Save_data.ipynb

def extract_custar_from_dir(work_dir):
    """
    Extract the value of c/ustar from the file name.
    """

    match = re.search(r"custar(\d+)", work_dir)
    if match:
        return match.group(1)
    return "Unknown"


def extract_direction_from_dir(work_dir):
    """
    Extract wind direction from the file name.
    """
    match = re.search(r"(forward|backward)", work_dir)
    if match:
        return match.group(1)
    return "Unknown"

def process_directory(work_dir, L0, N, tot_row, k_, mu2):
    custar_suffix = extract_custar_from_dir(work_dir)
    # custar_suffix = os.path.basename(os.path.normpath(work_dir)).split("custar")[1]
    # direction = os.path.basename(os.path.normpath(work_dir)) # 'forward' o 'backward'
    direction = extract_direction_from_dir(work_dir)

    time_fld = pd.read_csv(work_dir + "field/log_field.out", header=None, sep=" ")
    time_fld = time_fld.to_numpy()

    time_eta = pd.read_csv(work_dir + "eta/global_int.out", header=None, sep=" ")
    time_eta = time_eta.to_numpy()

    x_int = np.linspace(-L0 / 2, L0 / 2, N, endpoint=False) + L0 / N / 2
    z_int = np.linspace(-L0 / 2, L0 / 2, N, endpoint=False) + L0 / N / 2
    x_til, z_til = np.meshgrid(x_int, z_int)

    for i in range(len(time_fld)):
        #
        # define the time and row
        #
        time = time_fld[i, 0]
        istep = int(time_fld[i, 1])
        istep_c = f"{istep:09d}"
        print("****************")
        print(istep, time, i)
        print("****************")
        #
        # load eta_loc
        #
        etalo = np.fromfile(work_dir + "eta/eta_loc/eta_loc_t" + istep_c + ".bin")
        size = etalo.shape
        tot_row_i = int(size[0] / tot_row)
        print(tot_row_i)
        etalo = etalo.reshape([tot_row_i, tot_row])
        #
        # we remove bubbles for interpolate interface
        #
        print("First pass of remove")
        eta_m0 = 1.0
        cirp_th = 0.20
        new_row = 0
        for i in range(tot_row_i):
            if abs(etalo[i][12] - eta_m0) < cirp_th:
                new_row += 1
        #
        print("Second pass of remove")
        etal = np.zeros([new_row, 18])
        for i in range(new_row):
            if abs(etalo[i][12] - eta_m0) < cirp_th:
                etal[i][:] = etalo[i][:]
        #
        print("Assign array")
        xpo = etal[:, 0]
        zpo = etal[:, 1]
        pre = etal[:, 2]
        Sxx = etal[:, 3]
        Syy = etal[:, 4]
        Szz = etal[:, 5]
        Sxy = etal[:, 6]
        Sxz = etal[:, 7]
        Syz = etal[:, 8]
        uxi = etal[:, 9]
        uyi = etal[:, 10]
        uzi = etal[:, 11]
        eta = etal[:, 12]
        eps = etal[:, 13]
        n_x = etal[:, 14]
        n_y = etal[:, 15]
        n_z = etal[:, 16]
        #
        # we interpolate eta on a 2D cartesian grid with equidistant spacing equal to the printing resolution (2**9)
        #
        print("Interpolation to a Cartesian grid")
        pre_int = interp_2d(xpo, zpo, x_til, z_til, pre)
        Sxx_int = interp_2d(xpo, zpo, x_til, z_til, Sxx)
        Syy_int = interp_2d(xpo, zpo, x_til, z_til, Syy)
        Szz_int = interp_2d(xpo, zpo, x_til, z_til, Szz)
        Sxy_int = interp_2d(xpo, zpo, x_til, z_til, Sxy)
        Sxz_int = interp_2d(xpo, zpo, x_til, z_til, Sxz)
        Syz_int = interp_2d(xpo, zpo, x_til, z_til, Syz)
        uxi_int = interp_2d(xpo, zpo, x_til, z_til, uxi)
        uyi_int = interp_2d(xpo, zpo, x_til, z_til, uyi)
        uzi_int = interp_2d(xpo, zpo, x_til, z_til, uzi)
        eta_int = interp_2d(xpo, zpo, x_til, z_til, eta)
        eps_int = interp_2d(xpo, zpo, x_til, z_til, eps)
        n_x_int = interp_2d(xpo, zpo, x_til, z_til, n_x)
        n_y_int = interp_2d(xpo, zpo, x_til, z_til, n_y)
        n_z_int = interp_2d(xpo, zpo, x_til, z_til, n_z)
        #
        # compute momentum and energy fluxes
        #
        print("Compute momentum flux - pressure")
        [mf_px, mf_py, mf_pz] = mom_flux_p(pre_int, n_x_int, n_y_int, n_z_int)
        print("Compute momentum flux - viscous dissipation")
        [mf_vx, mf_vy, mf_vz] = mom_flux_v(
            Sxx_int,
            Sxy_int,
            Sxz_int,
            Syy_int,
            Syz_int,
            Szz_int,
            n_x_int,
            n_y_int,
            n_z_int,
            mu2,
        )
        print("Energy flux - pressure")
        en_p = ene_flux_p(pre_int, uxi_int, uyi_int, uzi_int, n_x_int, n_y_int, n_z_int)
        print("Energy flux - viscous dissipation")
        en_v = ene_flux_v(
            Sxx_int,
            Sxy_int,
            Sxz_int,
            Syy_int,
            Syz_int,
            Szz_int,
            uxi_int,
            uyi_int,
            uzi_int,
            n_x_int,
            n_y_int,
            n_z_int,
            mu2,
        )
        #
        # compute stress budget (pressure and viscous term)
        #
        eta_1d = np.average(eta_int, axis=0) - np.average(eta)
        pre_1d = np.average(pre_int, axis=0) - np.average(pre)
        Sxx_1d = np.average(Sxx_int, axis=0)
        Sxy_1d = np.average(Sxy_int, axis=0)
        mf_px_alt = mom_flux_p_alt(pre_1d, eta_1d, L0, N)
        mf_vx_alt = mom_flux_v_alt(Sxx_1d, Sxy_1d, eta_1d, L0, N, mu2)
        #
        # compute amplitude
        #
        ak = get_amp(eta_int, k_, L0)
        #
        # load the 2d span-averaged binary files
        #
        print("Load field and phase partition")
        fv_2d = return_file(work_dir, "fv", istep_c, N, 1)
        ux_2d = return_file(work_dir, "ux", istep_c, N, 1)
        uy_2d = return_file(work_dir, "uy", istep_c, N, 1)
        pr_2d = return_file(work_dir, "pr", istep_c, N, 1)
        di_2d = return_file(work_dir, "di", istep_c, N, 1)
        [ux_2d_air, ux_2d_wat, uy_2d_air, uy_2d_wat] = phase_partion(
            ux_2d, uy_2d, fv_2d, 1.0, 1.0
        )
        [pr_2d_air, pr_2d_wat, di_2d_air, di_2d_wat] = phase_partion(
            pr_2d, di_2d, fv_2d, 1.0, 1.0
        )
        #
        eta_m0 = 1
        print("From cartesian to wf")
        [ux_air_2d_wf, ux_air_1d_wf, zplot_air, zeta_air] = cart_to_wf(
            ux_2d_air, eta_1d, N, L0, k_, eta_m0
        )
        # ux_air
        [ux_wat_2d_wf, ux_wat_1d_wf, zplot_wat, zeta_wat] = cart_to_wf(
            ux_2d_wat, eta_1d, N, L0, k_, eta_m0
        )
        # ux_wat
        [pr_air_2d_wf, pr_air_1d_wf, zplot_air, zeta_air] = cart_to_wf(
            pr_2d_air, eta_1d, N, L0, k_, eta_m0
        )
        # pr_air (we do not need the one in water)
        [di_air_2d_wf, di_air_1d_wf, zplot_air, zeta_air] = cart_to_wf(
            di_2d_air, eta_1d, N, L0, k_, eta_m0
        )
        # di_air
        [di_wat_2d_wf, di_wat_1d_wf, zplot_wat, zeta_wat] = cart_to_wf(
            di_2d_wat, eta_1d, N, L0, k_, eta_m0
        )
        # di_wat
        #

        print("Final print of glo_obs")
        f = open(f"glo_obs_post__{direction}_{custar_suffix}.out", "a")
        f.write(
            "%.15f %.15f %.15f %.15f %.15f %.15f %.15f %.15f \n"
            % (1.0 * istep, time, ak, mf_px, mf_py, mf_pz, en_p, en_v)
        )
        f.flush()
        f.close()

        print("Final print of glo_obs alt")
        f = open(f"glo_obs_post_alt__{direction}_{custar_suffix}.out", "a")
        f.write(
            "%.15f %.15f %.15f %.15f \n" % (1.0 * istep, time, mf_px_alt, mf_vx_alt)
        )
        f.flush()
        f.close()

        print("Final print of wf")
        # Crear la carpeta si no existe
        folder_name = f"wave_coord_{direction}_{custar_suffix}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Escribir el archivo dentro de la carpeta

        filename = os.path.join(
            folder_name, f"prof_wf_{direction}_{custar_suffix}{istep_c}.out"
        )
        f = open(filename, "w")
        for i in range(N):
            f.write(
                "%.15f %.15f %.15f %.15f %.15f %.15f %.15f\n"
                % (
                    zeta_air[i],
                    zeta_wat[i],
                    ux_air_1d_wf[i],
                    ux_wat_1d_wf[i],
                    pr_air_1d_wf[i],
                    di_wat_1d_wf[i],
                    di_air_1d_wf[i],
                )
            )
        f.flush()
        f.close()