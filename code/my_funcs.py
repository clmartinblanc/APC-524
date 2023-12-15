import os
import re
import numpy as np
import math as m
import matplotlib.pyplot as plt
import pandas as pd
import gc
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import signal, interpolate
import sys
from tqdm import tqdm


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
        # the mean has been aly subtracted
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

    def interp_2d(xdata, zdata, xtile, ztile, fld):
        fld_int = griddata(
            (xdata.ravel(), zdata.ravel()),
            fld.ravel(),
            (xtile, ztile),
            method="nearest",
        )
        return fld_int


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
        self.variance.append(varr)
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

    def spectrum_integration_3d(self, data, L0, N, T, time):
        dt = time[1] - time[0]  # sampling intervals, (s)
        T = time[-1] - time[0]  # total duration
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
            return self.spectrum_integration_3d(data, L0, N, T, time)
        else:
            raise ValueError("Dimension must be '2D' or '3D'")


def phase_partion(ux, uy, f, s1_exp, s2_exp):
    ux_air = ux * (1.0 - f) ** s1_exp
    ux_water = ux * f ** (1 / s2_exp)
    uy_air = uy * (1.0 - f) ** s1_exp
    uy_water = uy * f ** (1 / s2_exp)
    return ux_air, ux_water, uy_air, uy_water


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


def exponential_func(x, a, b):
    return a * np.exp(b * x)


# for Save_data.ipynb


class DataProcessor:
    def __init__(
        self,
        work_dir,
        L0=2 * np.pi,
        N=512,
        tot_row=18,
        k_=4,
        u_s=0.25,
        rho_r=1000 / 1.225,
        Re_tau=720,
        h=1,
    ):
        self.work_dir = work_dir
        self.L0 = L0
        self.N = N
        self.tot_row = tot_row
        self.k_ = k_
        self.u_s = u_s
        self.rho_r = rho_r
        self.Re_tau = Re_tau
        self.h = h
        self.rho1 = 1
        self.rho2 = self.rho1 / self.rho_r
        self.mu2 = self.rho2 * self.u_s * (self.L0 - self.h) / self.Re_tau
        self.data = None
        self.eta_series = None

    def extract_custar_from_dir(self):
        """
        Extract the value of c/ustar from the file name.
        """

        match = re.search(r"custar(\d+)", self.work_dir)
        return match.group(1) if match else "Unknown"

    def extract_direction_from_dir(self):
        """
        Extract wind direction from the file name.
        """
        match = re.search(r"(forward|backward)", self.work_dir)
        return match.group(1) if match else "Unknown"

    def process_data(self):
        text_data = np.loadtxt(os.path.join(self.work_dir, "eta/global_int.out"))
        self.load_global_integrals()
        self.process_eta_data()
        self.process_velocity_and_pressure_data()

    def load_global_integrals(self):
        data = np.loadtxt(os.path.join(self.work_dir, "eta/global_int.out"))
        self.istep_c, self.time = data[:, 1], data[:, 0]

    def process_eta_data(self):
        self.eta_series = np.zeros(
            (len(self.istep_c), self.N, self.N), dtype=np.float32
        )
        for j, i in enumerate(self.istep_c):
            etalo = np.fromfile(
                os.path.join(self.work_dir, f"eta/eta_loc/eta_loc_t0000{int(i)}.bin")
            ).reshape([-1, self.tot_row])

            # Filtrado de datos
            new_row_count = sum(np.abs(etalo[:, 12] - 1.0) < 0.20)
            etal = np.zeros([new_row_count, 18])
            row_idx = 0
            for row in etalo:
                if np.abs(row[12] - 1.0) < 0.20:
                    etal[row_idx] = row
                    row_idx += 1

            xarray = np.linspace(
                -self.L0 / 2, self.L0 / 2, self.N, endpoint=False
            ) + self.L0 / (2 * self.N)
            xtile, ytile = np.meshgrid(xarray, xarray)
            eta = griddata(
                (etal[:, 0], etal[:, 1]), etal[:, 12], (xtile, ytile), method="nearest"
            )
            self.eta_series[j] = eta

    def process_velocity_and_pressure_data(self):
        data_list = []
        for i_int, t in zip(self.istep_c, self.time):
            data_dict = self.process_individual_step(i_int, t)
            data_list.append(data_dict)
        self.data = pd.DataFrame(data_list)
        self.data["eta"] = list(self.eta_series)

    def process_individual_step(self, i_int, t):
        data_dict = {"i": i_int, "t": t}
        patterns = [
            "ux_2d_avg_{:09d}.bin",
            "uy_2d_avg_{:09d}.bin",
            "uz_2d_avg_{:09d}.bin",
            "fv_2d_avg_{:09d}.bin",
            "pr_2d_avg_{:09d}.bin",
        ]
        for idx, pattern in enumerate(patterns):
            filename = pattern.format(int(i_int))
            file_path = os.path.join(self.work_dir + "field/", filename)
            if os.path.exists(file_path):
                array_data = np.fromfile(file_path).reshape((self.N, self.N))
                key = pattern.split("_")[0]
                data_dict[key] = array_data
                data_dict[key + "_mean"] = np.average(array_data, axis=0)
        return data_dict

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

    def process_directory(self):
        custar_suffix = self.extract_custar_from_dir(self.work_dir)
        # custar_suffix = os.path.basename(os.path.normpath(work_dir)).split("custar")[1]
        # direction = os.path.basename(os.path.normpath(work_dir)) # 'forward' o 'backward'
        direction = self.extract_direction_from_dir(self.work_dir)

        time_fld = pd.read_csv(
            self.work_dir + "field/log_field.out", header=None, sep=" "
        )
        time_fld = time_fld.to_numpy()

        time_eta = pd.read_csv(
            self.work_dir + "eta/global_int.out", header=None, sep=" "
        )
        time_eta = time_eta.to_numpy()

        x_int = (
            np.linspace(-self.L0 / 2, self.L0 / 2, self.N, endpoint=False)
            + self.L0 / self.N / 2
        )
        z_int = (
            np.linspace(-self.L0 / 2, self.L0 / 2, self.N, endpoint=False)
            + self.L0 / self.N / 2
        )
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
            etalo = np.fromfile(
                self.work_dir + "eta/eta_loc/eta_loc_t" + istep_c + ".bin"
            )
            size = etalo.shape
            tot_row_i = int(size[0] / self.tot_row)
            print(tot_row_i)
            etalo = etalo.reshape([tot_row_i, self.tot_row])
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
            pre_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, pre)
            Sxx_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, Sxx)
            Syy_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, Syy)
            Szz_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, Szz)
            Sxy_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, Sxy)
            Sxz_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, Sxz)
            Syz_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, Syz)
            uxi_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, uxi)
            uyi_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, uyi)
            uzi_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, uzi)
            eta_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, eta)
            eps_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, eps)
            n_x_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, n_x)
            n_y_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, n_y)
            n_z_int = CoordinateConverter.interp_2d(xpo, zpo, x_til, z_til, n_z)
            #
            # compute momentum and energy fluxes
            #
            print("Compute momentum flux - pressure")
            [mf_px, mf_py, mf_pz] = Utilities.mom_flux_p(
                pre_int, n_x_int, n_y_int, n_z_int
            )
            print("Compute momentum flux - viscous dissipation")
            [mf_vx, mf_vy, mf_vz] = Utilities.mom_flux_v(
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
            en_p = Utilities.ene_flux_p(
                pre_int, uxi_int, uyi_int, uzi_int, n_x_int, n_y_int, n_z_int
            )
            print("Energy flux - viscous dissipation")
            en_v = Utilities.ene_flux_v(
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
                self.mu2,
            )
            #
            # compute stress budget (pressure and viscous term)
            #
            eta_1d = np.average(eta_int, axis=0) - np.average(eta)
            pre_1d = np.average(pre_int, axis=0) - np.average(pre)
            Sxx_1d = np.average(Sxx_int, axis=0)
            Sxy_1d = np.average(Sxy_int, axis=0)
            mf_px_alt = Utilities.mom_flux_p_alt(pre_1d, eta_1d, self.L0, self.N)
            mf_vx_alt = Utilities.mom_flux_v_alt(
                Sxx_1d, Sxy_1d, eta_1d, self.L0, self.N, self.mu2
            )
            #
            # compute amplitude
            #
            ak = Utilities.get_amp(eta_int, k_, L0)
            #
            # load the 2d span-averaged binary files
            #
            print("Load field and phase partition")
            fv_2d = FileHandler.return_file(self.work_dir, "fv", istep_c, self.N, 1)
            ux_2d = FileHandler.return_file(self.work_dir, "ux", istep_c, self.N, 1)
            uy_2d = FileHandler.return_file(self.work_dir, "uy", istep_c, self.N, 1)
            pr_2d = FileHandler.return_file(self.work_dir, "pr", istep_c, self.N, 1)
            di_2d = FileHandler.return_file(self.work_dir, "di", istep_c, self.N, 1)
            [ux_2d_air, ux_2d_wat, uy_2d_air, uy_2d_wat] = phase_partion(
                ux_2d, uy_2d, fv_2d, 1.0, 1.0
            )
            [pr_2d_air, pr_2d_wat, di_2d_air, di_2d_wat] = phase_partion(
                pr_2d, di_2d, fv_2d, 1.0, 1.0
            )
            #
            eta_m0 = 1
            print("From cartesian to wf")
            [
                ux_air_2d_wf,
                ux_air_1d_wf,
                zplot_air,
                zeta_air,
            ] = CoordinateConverter.cart_to_wf(
                ux_2d_air, eta_1d, self.N, self.L0, self.k_, eta_m0
            )
            # ux_air
            [
                ux_wat_2d_wf,
                ux_wat_1d_wf,
                zplot_wat,
                zeta_wat,
            ] = CoordinateConverter.cart_to_wf(
                ux_2d_wat, eta_1d, self.N, self.L0, self.k_, eta_m0
            )
            # ux_wat
            [
                pr_air_2d_wf,
                pr_air_1d_wf,
                zplot_air,
                zeta_air,
            ] = CoordinateConverter.cart_to_wf(
                pr_2d_air, eta_1d, self.N, self.L0, self.k_, eta_m0
            )
            # pr_air (we do not need the one in water)
            [
                di_air_2d_wf,
                di_air_1d_wf,
                zplot_air,
                zeta_air,
            ] = CoordinateConverter.cart_to_wf(
                di_2d_air, eta_1d, self.N, self.L0, self.k_, eta_m0
            )
            # di_air
            [
                di_wat_2d_wf,
                di_wat_1d_wf,
                zplot_wat,
                zeta_wat,
            ] = CoordinateConverter.cart_to_wf(
                di_2d_wat, eta_1d, self.N, self.L0, self.k_, eta_m0
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
                f"{1.0 * istep:.15f} {time:.15f} {mf_px_alt:.15f} {mf_vx_alt:.15f} \n"
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
            pass


# for Graphs.ipynb


class DataPlotter:
    def __init__(self):
        pass

    # Define different symbols for forward and backward directions
    markers = {"forward": "o", "backward": "x"}

    def darken_color(color, factor=0.7):
        """Darkens a given hexadecimal color."""
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def plot_data_color(
        self, ax, istep, time, direction, color, marker, custar_suffix, data_type
    ):
        """
        Plot data with specific color and marker based on direction, custar_suffix, and data type (air or water).

        Parameters:
        - ax (Axes): Axes object for plotting.
        - istep (iterable): Time steps for data files.
        - time (iterable): Corresponding time values.
        - direction (str): Direction part of the filename.
        - color (str): Color for the plot.
        - marker (str): Marker style for the plot.
        - custar_suffix (str): Suffix for the filename.
        - data_type (str): Type of data to plot ('air' or 'water').
        """
        for i, t in zip(istep, time):
            formatted_i = f"{int(i):09d}"
            filename_wf = f"wave_coord_{direction}_{custar_suffix}/prof_wf_{direction}_{custar_suffix}{formatted_i}.out"
            if os.path.exists(filename_wf):
                data_wf = np.loadtxt(filename_wf)

                if data_type == "air":
                    zeta = data_wf[:, 0]
                    ux_1d_wf = data_wf[:, 2]
                elif data_type == "water":
                    zeta = data_wf[:, 1]
                    ux_1d_wf = data_wf[:, 3]
                else:
                    raise ValueError("Invalid data type. Choose 'air' or 'water'.")

                label = f"{direction} custar {custar_suffix} at time {t}"
                # Oscurece el color si es el último tiempo
                if t == time[-1]:
                    color = self.darken_color(color)
                ax.plot(
                    zeta,
                    ux_1d_wf,
                    color=color,
                    marker=marker,
                    markersize=7,
                    label=label,
                    linewidth=0.5,
                    alpha=0.5,
                )

    def plot_data(
        self, ax, istep, time, direction, cmap, norm, custar_suffix, data_type
    ):
        """
        Plots data (air or water properties) for various time steps on a matplotlib Axes.

        Parameters:
        - ax (matplotlib.axes.Axes): Axes object for plotting.
        - istep (iterable): Time steps for data files.
        - time (iterable): Corresponding time values for color mapping.
        - direction (str): Direction part of the filename.
        - cmap (Colormap): Colormap for plot colors.
        - norm (Normalize): Normalization for time values.
        - custar_suffix (str): Suffix for the filename.
        - data_type (str): Type of data to plot ('air' or 'water').
        """
        for i, t in zip(istep, time):
            formatted_i = f"{int(i):09d}"
            filename_wf = f"wave_coord_{direction}_{custar_suffix}/prof_wf_{direction}_{custar_suffix}{formatted_i}.out"
            if os.path.exists(filename_wf):
                data_wf = np.loadtxt(filename_wf)

                if data_type == "air":
                    zeta = data_wf[:, 0]
                    ux_1d_wf = data_wf[:, 2]
                elif data_type == "water":
                    zeta = data_wf[:, 1]
                    ux_1d_wf = data_wf[:, 3]
                else:
                    raise ValueError("Invalid data type. Choose 'air' or 'water'.")

                ax.plot(zeta, ux_1d_wf, color=cmap(norm(t)))

    def plot_data_color(
        self, ax, istep, time, direction, color, marker, custar_suffix, data_type
    ):
        """
        Plot data with specific color and marker based on direction, custar_suffix, and data type (air or water).

        Parameters:
        - ax (matplotlib.axes.Axes): Axes object for plotting.
        - istep (iterable): Time steps for data files.
        - time (iterable): Corresponding time values.
        - direction (str): Wind direction ('forward' or 'backward').
        - color (str): Color for the plot.
        - marker (str): Marker style for the plot.
        - custar_suffix (str): Suffix indicating specific c/ustar value.
        - data_type (str): Type of data to plot ('air' or 'water').
        """
        for i, t in zip(istep, time):
            formatted_i = f"{int(i):09d}"
            filename_wf = f"wave_coord_{direction}_{custar_suffix}/prof_wf_{direction}_{custar_suffix}{formatted_i}.out"
            if os.path.exists(filename_wf):
                data_wf = np.loadtxt(filename_wf)

                if data_type == "air":
                    zeta = data_wf[:, 0]
                    ux_1d_wf = data_wf[:, 2]
                elif data_type == "water":
                    zeta = data_wf[:, 1]
                    ux_1d_wf = data_wf[:, 3]
                else:
                    raise ValueError("Invalid data type. Choose 'air' or 'water'.")

                label = f"{direction} custar {custar_suffix} at time {t}"
                if t == time[-1]:
                    color = self.darken_color(color)
                ax.plot(
                    zeta,
                    ux_1d_wf,
                    color=color,
                    marker=marker,
                    markersize=7,
                    label=label,
                    linewidth=0.5,
                    alpha=0.5,
                )

    def process_and_plot(work_dir, ax, cmap_name):
        custar_value = self.extract_custar_from_dir(work_dir)
        ax.set_title(f"c/ustar = {custar_value}")

        df = self.process_data(work_dir)

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
