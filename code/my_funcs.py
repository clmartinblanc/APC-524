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

# common for all cases
L0 = 2.0 * np.pi
h = 1
u_s = 0.25
N = 512
k_ = 4
rho_r = 1000 / 1.225
Re_tau = 720
rho1 = 1
rho2 = rho1 / rho_r
mu2 = rho2 * u_s * (L0 - h) / Re_tau
tot_row = 18


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
        f.write(f"{1.0 * istep:.15f} {time:.15f} {mf_px_alt:.15f} {mf_vx_alt:.15f} \n")
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


# for Graphs.ipynb


def read_files_to_dfs(direction, custar_suffix):
    """
    Reads data from specified files and transforms it into Pandas DataFrames.

    This function reads two types of files: 'glo_obs_post__' and 'glo_obs_post_alt__',
    both of which are expected to have a specific suffix and direction in their filenames.
    The data from these files is loaded into numpy arrays and then converted into Pandas DataFrames.
    Each DataFrame is structured with specific columns based on the file content.

    Parameters:
    - direction (str): A string parameter that specifies the direction of the wind (forward or backward).
    - custar_suffix (str): A string parameter that defines the custom c/u* (2,4 or 8).

    Returns:
    - df_glo_obs (DataFrame): A DataFrame containing data from the 'glo_obs_post__' file.
      Columns include 'istep', 'time', 'ak', 'mf_px', 'mf_py', 'mf_pz', 'en_p', 'en_v'.
      The 'time' column is adjusted to start from 0.
    - df_glo_obs_alt (DataFrame): A DataFrame containing data from the 'glo_obs_post_alt__' file.
      Columns include 'istep', 'time', 'mf_px_alt', 'mf_vx_alt'.
      Similar to df_glo_obs, the 'time' column in this DataFrame is also adjusted to start from 0, it doesn't start in 0 directly because we use a precursor for the turbulence

    """

    # Reading the glo_obs file
    filename_glo = f"glo_obs_post__{direction}_{custar_suffix}.out"
    data_glo_obs = np.loadtxt(filename_glo)
    df_glo_obs = pd.DataFrame(
        data_glo_obs,
        columns=["istep", "time", "ak", "mf_px", "mf_py", "mf_pz", "en_p", "en_v"],
    )

    # Adjust the 'time' column to start from 0
    df_glo_obs["time"] = df_glo_obs["time"] - df_glo_obs["time"].iloc[0]

    # Reading the glo_obs_alt file
    filename_glo_alt = f"glo_obs_post_alt__{direction}_{custar_suffix}.out"
    data_glo_obs_alt = np.loadtxt(filename_glo_alt)
    df_glo_obs_alt = pd.DataFrame(
        data_glo_obs_alt, columns=["istep", "time", "mf_px_alt", "mf_vx_alt"]
    )

    # Adjust the 'time' column in the alt DataFrame to start from 0
    df_glo_obs_alt["time"] = df_glo_obs_alt["time"] - df_glo_obs_alt["time"].iloc[0]

    return df_glo_obs, df_glo_obs_alt


def plot_data_air(ax, istep, time, direction, cmap, norm, custar_suffix):
    """
    Plots air property data for various time steps on a matplotlib Axes.

    Parameters:
    - ax (matplotlib.axes.Axes): Axes object for plotting.
    - istep (iterable): Time steps for data files.
    - time (iterable): Corresponding time values for color mapping.
    - direction (str): Direction part of the filename.
    - cmap (Colormap): Colormap for plot colors.
    - norm (Normalize): Normalization for time values.
    - custar_suffix (str): Suffix for the filename.

    The function reads data from specified files, extracts air properties,
    and plots them on 'ax'. Each time step's data is colored using 'cmap'
    and 'norm' based on the corresponding time value.
    """
    for i, t in zip(istep, time):
        formatted_i = f"{int(i):09d}"
        filename_wf = f"wave_coord_{direction}_{custar_suffix}/prof_wf_{direction}_{custar_suffix}{formatted_i}.out"
        if os.path.exists(filename_wf):
            data_wf = np.loadtxt(filename_wf)
            zeta_air = data_wf[:, 0]
            ux_air_1d_wf = data_wf[:, 2]
            ax.plot(zeta_air, ux_air_1d_wf, color=cmap(norm(t)))


def darken_color(color, factor=0.7):
    """Darkens a given hexadecimal color."""
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_data_air_color(ax, istep, time, direction, color, marker, custar_suffix):
    """Plot data with specific color and marker based on direction and custar_suffix."""
    for i, t in zip(istep, time):
        formatted_i = f"{int(i):09d}"
        filename_wf = f"wave_coord_{direction}_{custar_suffix}/prof_wf_{direction}_{custar_suffix}{formatted_i}.out"
        if os.path.exists(filename_wf):
            data_wf = np.loadtxt(filename_wf)
            zeta_air = data_wf[:, 0]
            ux_air_1d_wf = data_wf[:, 2]
            label = f"{direction} custar {custar_suffix} at time {t}"
            # Darken the color for the last time step
            if t == time[-1]:
                color = darken_color(color)
            ax.plot(
                zeta_air,
                ux_air_1d_wf,
                color=color,
                marker=marker,
                markersize=7,
                label=label,
                linewidth=0.5,
                alpha=0.5,
            )


# Define different symbols for forward and backward directions
markers = {"forward": "o", "backward": "x"}


# Function to plot water data on a specific axis
def plot_water_data(ax, istep, time, direction, cmap, norm, suffix):
    for i, t in zip(istep, time):
        formatted_i = f"{int(i):09d}"  # Format the timestep for file naming
        # Constructing the filename based on direction and suffix
        filename_wf = f"wave_coord_{direction}_{suffix}/prof_wf_{direction}_{suffix}{formatted_i}.out"
        # Load the data from the file
        data_wf = np.loadtxt(filename_wf)
        # Extracting water spatial coordinate and property
        zeta_water = data_wf[:, 1]
        ux_water_1d_wf = data_wf[:, 3]
        # Plot the data on the provided Axes
        ax.plot(zeta_water, ux_water_1d_wf, color=cmap(norm(t)))


def darken_color(color, factor=0.7):
    """Oscurece un color dado en formato hexadecimal."""
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_data_water_color(ax, istep, time, direction, color, marker, custar_suffix):
    for i, t in zip(istep, time):
        formatted_i = f"{int(i):09d}"
        filename_wf = f"wave_coord_{direction}_{custar_suffix}/prof_wf_{direction}_{custar_suffix}{formatted_i}.out"
        if os.path.exists(filename_wf):
            data_wf = np.loadtxt(filename_wf)
            zeta_air = data_wf[:, 1]
            ux_air_1d_wf = data_wf[:, 3]
            label = f"{direction} custar {custar_suffix} at time {t}"
            # Oscurece el color si es el último tiempo
            if t == time[-1]:
                color = darken_color(color)
            ax.plot(
                zeta_air,
                ux_air_1d_wf,
                color=color,
                marker=marker,
                markersize=7,
                label=label,
                linewidth=0.5,
                alpha=0.5,
            )


# for postprocessing.ipynb


def process_data(work_dir):
    N = 512
    L0 = 2 * np.pi

    # Leer el archivo de texto
    data = np.loadtxt(os.path.join(work_dir, "eta/global_int.out"))
    istep_c, time = data[:, 1], data[:, 0]

    eta_series = np.zeros((istep_c.shape[0], N, N), dtype=np.float32)

    for j, i in enumerate(istep_c):
        etalo = np.fromfile(
            os.path.join(work_dir, f"eta/eta_loc/eta_loc_t0000{int(i)}.bin")
        )
        etalo = etalo.reshape([int(etalo.size / 18), 18])

        etal = [row for row in etalo if abs(row[12] - 1.0) < 0.20]
        etal = np.array(etal)

        xarray = np.linspace(-L0 / 2, L0 / 2, N, endpoint=False) + L0 / (2 * N)
        yarray = xarray  # Son iguales en este contexto
        xtile, ytile = np.meshgrid(xarray, yarray)
        eta = griddata(
            (etal[:, 0].ravel(), etal[:, 1].ravel()),
            etal[:, 12].ravel(),
            (xtile, ytile),
            method="nearest",
        )

        eta_series[j] = eta

    patterns = [
        "ux_2d_avg_{:09d}.bin",
        "uy_2d_avg_{:09d}.bin",
        "uz_2d_avg_{:09d}.bin",
        "fv_2d_avg_{:09d}.bin",
        "pr_2d_avg_{:09d}.bin",
    ]
    data_list = []

    for i_int, t in zip(istep_c, time):
        data_dict = {"i": i_int, "t": t}

        for idx, pattern in enumerate(patterns):
            filename = pattern.format(int(i_int))
            file_path = os.path.join(work_dir + "field/", filename)

            if os.path.exists(file_path):
                array_data = np.fromfile(file_path)
                reshaped_data = array_data.reshape((N, N))

                if idx == 0:
                    data_dict["ux"] = reshaped_data
                    data_dict["ux_mean"] = np.average(reshaped_data, axis=0)
                elif idx == 1:
                    data_dict["uy"] = reshaped_data
                    data_dict["uy_mean"] = np.average(reshaped_data, axis=0)
                elif idx == 2:
                    data_dict["uz"] = reshaped_data
                    data_dict["uz_mean"] = np.average(reshaped_data, axis=0)
                elif idx == 3:
                    data_dict["fv"] = reshaped_data
                elif idx == 4:
                    data_dict["pressure"] = reshaped_data

        data_list.append(data_dict)

    return pd.DataFrame(data_list)


def process_and_plot(work_dir, ax, cmap_name):
    custar_value = extract_custar_from_dir(work_dir)
    ax.set_title(f"c/ustar = {custar_value}")

    df = process_data(work_dir)

    # Filtrar tiempos para reducir la superposición
    unique_times = df["t"].unique()
    sampled_times = unique_times[::5]

    cmap = plt.get_cmap(cmap_name, len(sampled_times))

    for idx, time in enumerate(sampled_times):
        df_time = df[df["t"] == time]
        ux_means = df_time["ux_mean"].values[0]
        ax.plot(y, ux_means / ustar, color=cmap(idx), lw=1.0, linestyle="-")

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap),
        ax=ax,
        orientation="vertical",
        fraction=0.05,
        pad=0.05,
    )
    cbar.set_label("Time", size=12)

    cbar.set_ticks(np.linspace(0, 1, len(sampled_times)))
    cbar.set_ticklabels([f"{time:.2f}" for time in sampled_times])
