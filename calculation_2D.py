"""
2D_calculation: import function to perform DBM simulation in 2D
Author: Ronish Mugatwala
E-mail: ronish.mugatwala@edu.unige.it
Github: astronish16

Updates:
19th June 2024: new project structure.

"""

import sys
from icecream import ic
import numpy as np
import scipy as sc
import scipy.stats as sts
from scipy.optimize import newton
from astroquery.jplhorizons import Horizons
from astropy.time import Time
import astropy.units as u
import ephem
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
from io import BytesIO
from PIL import Image

import dbm_functions as dbm


class DBM2D:
    def __init__(
        self,
        time_utc,
        r0,
        v0,
        Omega,
        Phi_CME,
        target_name,
        cone_type="IC",
        Kinematic="SSE",
        auto_dbm=True,
        P_DBM=True,
        wind_type="Slow",
        w=None,
        gamma=None,
        dt=None,
        dr0=None,
        dv0=None,
        dw=None,
        dgamma=None,
        domega=None,
        dphi_cme=None,
    ):
        """
        This method initialize the class and validate the inputs.

        Args:
            time_utc (datetime object): time correspond to CME postion at r0
            r0 (float): initial postion of CME [R⊙, generally beyond 20 solar radii]
            v0 (float): speed of CME at r0 [km/s]
            Omega (float): half angular width of CME cone [deg]
            Phi_CME (float): central meridian of CME in stornyhurst coordinate system [deg]
                             (Standard values from DONKI, Xie's Cone, Zaho's cone)
            target_name (str): keyword for planet/ spacecraft
            cone_type (str, optional): Type of cone geometry considered for CME. Defaults to "IC".
            Kinematic (str, optional): Type of kinematic approach considered during CME propagation. Defaults to "SSE".
            auto_dbm (bool, optional): Choice to select free DBM parameters manually or from Mugatwala et al 2024.
                                       Defaults to True.
            P_DBM (bool, optional): Simple DBM/ P-DBM. Defaults to True.
            wind_type (str, optional): selection of solar wind type in automatic selection of DBM free parameters.
                                       Defaults to "Slow".
            w (float, optional): value of ambient solar wind speed for manual entry. Defaults to None. [~ 400 km/s]
            gamma (float, optional): drag parameter. Defaults to None. [~0.2E-7 km-1]
            dt (float, optional): Uncertainity in the t0. Defaults to None. [min]
            dr0 (float, optional): Uncertanity in r0. Defaults to None. [R⊙]
            dv0 (float, optional): Uncertanity in v0. Defaults to None. [km/s]
            dw (float, optional): Uncertanity in w. Defaults to None. [~ 50 km/s]
            dgamma (float, optional): Uncertanity in gamma. Defaults to None. [~0.1E-7 km-1]
            domega (float, optional): Uncertanity in omega. Defaults to None. [deg]
            dphi_cme (float, optional): Uncertanity in phi_cme. Defaults to None. [deg]

        Raises:
            ValueError: None value of W and gamma in case of auto_dbm = False.
        """
        self.time_utc = time_utc
        self.T0 = time_utc.timestamp()
        self.r0 = ((r0 * u.R_sun).to(u.km)).value
        self.v0 = v0
        self.target_name = target_name
        self.Phi_target, self.r_target = dbm.position(target_name, time_utc)
        self.r1 = ((self.r_target * u.au).to(u.km)).value
        ic(self.r1, self.Phi_target)

        self.Omega = Omega
        self.Phi_CME = dbm.Phi_Correction(Phi_CME, time_utc)
        ic(self.Phi_CME)

        self.auto_dbm = auto_dbm
        self.wind_type = wind_type
        self.P_DBM = P_DBM
        self.N = 10000

        if self.auto_dbm == False:
            if w is None or gamma is None:
                raise ValueError(
                    "‼️ Input problem: 'w' and 'gamma' must be provided when auto_dbm is False."
                )
            self.w = w  # manual solar wind speed in km/s
            self.gamma = (
                gamma  # manual drag parameter in unit of km-1. (typical value: 0.2e-7)
            )

        if self.P_DBM == True:
            self.dt0 = dt * 60.0
            self.dr0 = (
                ((dr0 if dr0 is not None else 0.1 * r0) * u.R_sun).to(u.km)
            ).value
            self.dv0 = dv0 if dv0 is not None else 0.1 * self.v0
            self.domega = domega if domega is not None else 0.1 * self.Omega
            self.dphi_cme = dphi_cme if dphi_cme is not None else 0.1 * Phi_CME
            if self.auto_dbm == False:
                try:
                    ic()
                    self.dw = dw if dw is not None else 0.1 * self.w
                    self.dgamma = dgamma if dgamma is not None else 0.1 * self.gamma
                except Exception as e:
                    print("+" * 20)
                    print(f"‼️ Input problem...")
                    print(f"Problem with dw and dgamma valse due to {e}.")
                    print("+" * 20)

        valid_cones = [
            "IC",
            "Ice-Cream Cone",
            "TC",
            "Tangential Cone",
            "CC",
            "Concentric Cone",
        ]
        valid_kinematic = [
            "SSE",
            "Self-Simmilar Expansion",
            "FCE",
            "Flattening Cone Evolution",
        ]

        # Validate Cone Geometry
        if cone_type is None or cone_type not in valid_cones:
            print("+" * 20)
            print(f"Warning: Invalid Cone Geometry: {cone_geometry}.\n")
            print(f"Defaulting to IC")
            print("+" * 20)
            self.cone_type = "IC"
        else:
            self.cone_type = cone_type

        if Kinematic is None or Kinematic not in valid_kinematic:
            print("+" * 20)
            print(f"Warning: Invalid kinematic: {Kinematic}.\n")
            print(f"Defaulting to SSE")
            print("+" * 20)
            self.kinematic = "SSE"
        else:
            self.kinematic = Kinematic

        # Validating Omega for cone type
        if self.cone_type in ["IC", "Ice-Cream Cone"] and self.Omega > 90.0:
            print("=" * 20)
            print(
                f"Warning: CME cone type with provided angular width is not possible.\n Hence, Concentric cone is considered for calculation."
            )
            self.cone_type = "CC"
        elif self.cone_type in ["TC", "Tangential Cone"] and self.Omega >= 90.0:
            print("=" * 20)
            print(
                f"Warning: CME cone type with provided angular width is not possible.\n Hence, Concentric cone is considered for calculation."
            )
            self.cone_type = "CC"

    def P_DBM_run(self):
        """
        Main method of the class which perform (P)DBM simulation as per conditions provided.
        Returns:
            _dictionary_: dictionary containing all the essential information from the (P)DBM results.
                          i.e, transit time and speed (inclusing associated uncertainity), kinematic plots, DBM input etc...
        """

        if self.auto_dbm == True and self.P_DBM == True:
            w_array, gamma_array = dbm.auto_w_gamma_func(
                PDBM=self.P_DBM, wind_type=self.wind_type, N=self.N
            )
            ic("==")
            ic(self.auto_dbm, self.P_DBM)
            ic(self.Phi_CME)
            ic(self.Phi_target)
            ic(self.Omega)

            T_array, V_array, Hits, Miss, V0_array, Omega_array, Phi_CME_array = (
                dbm.PDBM_2D(
                    self.r0,
                    self.dr0,
                    self.r1,
                    self.v0,
                    self.dv0,
                    gamma_array,
                    w_array,
                    self.Omega,
                    self.domega,
                    self.Phi_CME,
                    self.dphi_cme,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                    self.dt0,
                    self.N,
                )
            )

            ic(T_array)
            ic(Hits)

            T_mean = np.nanmean(T_array)
            T_std = np.nanstd(T_array)
            V_mean = np.nanmean(V_array)
            V_std = np.nanstd(V_array)
            W_median = np.nanmedian(w_array)
            G_median = np.nanmedian(gamma_array)
            t_arrival = self.T0 + (np.nanmedian(T_array) * 3600)
            t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime("%Y-%m-%d %H:%M")
            tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            Probability_of_Arrival = Hits / self.N
            ic(t_arrival_UTC)

            # Making plot
            if Hits == 0:
                # Making plot
                rvt_plot = dbm.PDBM_RVT_plot(
                    self.time_utc,
                    T_array,
                    self.r0,
                    V0_array,
                    gamma_array,
                    w_array,
                    self.r_target,
                    tdate,
                )
                T_PDF_plot = dbm.TT_plot(T_array)
                V_PDF_Plot = dbm.V_plot(V_array)
            else:
                rvt_plot = dbm.PDBM_2D_RVT_plot(
                    self.time_utc,
                    T_array,
                    self.r0,
                    V0_array,
                    gamma_array,
                    w_array,
                    self.r_target,
                    tdate,
                    Omega_array,
                    Phi_CME_array,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                )
                T_PDF_plot = dbm.TT_plot(T_array)
                V_PDF_Plot = dbm.V_plot(V_array)

            cme_plot = dbm.setup_heliosphere(
                T_arrival_UTC,
                self.r_target,
                T_array,
                self.r0,
                V0_array,
                gamma_array,
                w_array,
                Omega_array,
                Phi_CME_array,
                self.Phi_target,
                self.cone_type,
                self.kinematic,
                self.P_DBM,
            )

            results = {
                "Transit_time_mean": T_mean,
                "Transit_time_std": T_std,
                "Arrival_speed_mean": V_mean,
                "Arrival_speed_std": V_std,
                "Arrival_time": t_arrival_UTC,
                "TT_distribution": T_PDF_plot,
                "V1_distribution": V_PDF_Plot,
                "RVT_plot": rvt_plot,
                "Travel_distance": self.r_target,
                "Initial_speed": self.v0,
                "w_median": W_median,
                "gamma_median": G_median,
                "Probability of Arrival": Probability_of_Arrival,
                "Heliosphere": cme_plot,
            }

        elif self.auto_dbm == True and self.P_DBM == False:
            w, gamma = dbm.auto_w_gamma_func(
                PDBM=self.P_DBM, wind_type=self.wind_type, N=1000
            )

            if (
                (self.Phi_CME - self.Omega)
                <= self.Phi_target
                <= (self.Phi_CME + self.Omega)
            ):
                T1, V1 = dbm.DBM_2D(
                    self.r0,
                    self.r1,
                    self.v0,
                    gamma,
                    w,
                    self.Omega,
                    self.Phi_CME,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                )
                Hit = "Shit!! CME hits the target. SW alert."
                t_arrival = self.T0 + (T1 * 3600)
                t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime(
                    "%Y-%m-%d %H:%M"
                )
                tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                ic(t_arrival_UTC)

                rvt_plot = dbm.DBM_2D_RVT_plot(
                    self.time_utc,
                    T1,
                    self.r0,
                    self.v0,
                    gamma,
                    w,
                    self.r_target,
                    tdate,
                    self.Omega,
                    self.Phi_CME,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                )
            else:
                # this is a case  when CME is not gonna hit the target so just regular 1D DBM calculation has been performed.
                T1, V1 = dbm.DBM(self.r0, self.r1, self.v0, gamma, w)
                Hit = "Yay, CME missed the target."
                t_arrival = self.T0 + (T1 * 3600)
                t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime(
                    "%Y-%m-%d %H:%M"
                )
                tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                ic(t_arrival_UTC)

                rvt_plot = dbm.DBM_RVT_plot(
                    self.time_utc, T1, self.r0, self.v0, gamma, w, self.r_target, tdate
                )

            # Getting Plots

            results = {
                "Transit_time_mean": T1,
                "Arrival_speed_mean": V1,
                "Arrival_time": t_arrival_UTC,
                "RVT_plot": rvt_plot,
                "Travel_distance": self.r_target,
                "Initial_speed": self.v0,
                "w_median": w,
                "gamma_median": gamma,
                "Hit": Hit,
            }

        elif self.auto_dbm == False and self.P_DBM == True:
            w_array = np.random.normal(self.w, self.dw, self.N)
            gamma_array = np.clip(
                np.random.normal(self.gamma, self.dgamma, self.N),
                1.0e-9,
                3.0e-7,
            )
            T_array, V_array, Hits, Miss, V0_array, Omega_array, Phi_CME_array = (
                dbm.PDBM_2D(
                    self.r0,
                    self.dr0,
                    self.r1,
                    self.v0,
                    self.dv0,
                    gamma_array,
                    w_array,
                    self.Omega,
                    self.domega,
                    self.Phi_CME,
                    self.dphi_cme,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                    self.dt0,
                    self.N,
                )
            )

            T_mean = np.nanmean(T_array)
            T_std = np.nanstd(T_array)
            V_mean = np.nanmean(V_array)
            V_std = np.nanstd(V_array)
            W_median = np.nanmedian(w_array)
            G_median = np.nanmedian(gamma_array)
            t_arrival = self.T0 + (np.nanmedian(T_array) * 3600)
            t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime("%Y-%m-%d %H:%M")
            tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            Probability_of_Arrival = Hits / self.N
            ic(t_arrival_UTC)

            # Making plot
            if Hits == 0:
                # Making plot
                rvt_plot = dbm.PDBM_RVT_plot(
                    self.time_utc,
                    T_array,
                    self.r0,
                    V0_array,
                    gamma_array,
                    w_array,
                    self.r_target,
                    tdate,
                )
                T_PDF_plot = dbm.TT_plot(T_array)
                V_PDF_Plot = dbm.V_plot(V_array)
            else:
                rvt_plot = dbm.PDBM_2D_RVT_plot(
                    self.time_utc,
                    T_array,
                    self.r0,
                    V0_array,
                    gamma_array,
                    w_array,
                    self.r_target,
                    tdate,
                    Omega_array,
                    Phi_CME_array,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                )
                T_PDF_plot = dbm.TT_plot(T_array)
                V_PDF_Plot = dbm.V_plot(V_array)

            results = {
                "Transit_time_mean": T_mean,
                "Transit_time_std": T_std,
                "Arrival_speed_mean": V_mean,
                "Arrival_speed_std": V_std,
                "Arrival_time": t_arrival_UTC,
                "TT_distribution": T_PDF_plot,
                "V1_distribution": V_PDF_Plot,
                "RVT_plot": rvt_plot,
                "Travel_distance": self.r_target,
                "Initial_speed": self.v0,
                "w_median": W_median,
                "gamma_median": G_median,
                "Probability of Arrival": Probability_of_Arrival,
            }

        elif self.auto_dbm == False and self.P_DBM == False:
            w = self.w
            gamma = self.gamma

            if (
                (self.Phi_CME - self.Omega)
                <= self.Phi_target
                <= (self.Phi_CME + self.Omega)
            ):
                T1, V1 = dbm.DBM_2D(
                    self.r0,
                    self.r1,
                    self.v0,
                    gamma,
                    w,
                    self.Omega,
                    self.Phi_CME,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                )
                Hit = "Shit!! CME hits the target. SW alert."
                t_arrival = self.T0 + (T1 * 3600)
                t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime(
                    "%Y-%m-%d %H:%M"
                )
                tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                ic(t_arrival_UTC)

                rvt_plot = dbm.DBM_2D_RVT_plot(
                    self.time_utc,
                    T1,
                    self.r0,
                    self.v0,
                    gamma,
                    w,
                    self.r_target,
                    tdate,
                    self.Omega,
                    self.Phi_CME,
                    self.Phi_target,
                    self.cone_type,
                    self.kinematic,
                )
            else:
                # this is a case  when CME is not gonna hit the target so just regular 1D DBM calculation has been performed.
                T1, V1 = dbm.DBM(self.r0, self.r1, self.v0, gamma, w)
                Hit = "Yay, CME missed the target."
                t_arrival = self.T0 + (T1 * 3600)
                t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime(
                    "%Y-%m-%d %H:%M"
                )
                tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
                ic(t_arrival_UTC)

                rvt_plot = dbm.DBM_RVT_plot(
                    self.time_utc, T1, self.r0, self.v0, gamma, w, self.r_target, tdate
                )

            # Getting Plots
            cme_plot = dbm.setup_heliosphere(
                T_arrival_UTC,
                self.r_target,
                T1,
                self.r0,
                self.v0,
                gamma,
                w,
                self.Omega,
                self.Phi_CME,
                self.Phi_target,
                self.cone_type,
                self.kinematic,
            )

            results = {
                "Transit_time_mean": T1,
                "Arrival_speed_mean": V1,
                "Arrival_time": t_arrival_UTC,
                "RVT_plot": rvt_plot,
                "Travel_distance": self.r_target,
                "Initial_speed": self.v0,
                "w_median": w,
                "gamma_median": gamma,
                "Hit": Hit,
                "Heliosphere": cme_plot,
            }

        return results


# ▶️ Example usage:
if __name__ == "__main__":
    downloader = DBM2D(
        time_utc=datetime(2022, 12, 1, 12, 23),
        r0=20,
        v0=1000,
        Omega=30,
        Phi_CME=10,
        cone_type="IC",
        Kinematic="SSE",
        target_name="Earth",
        P_DBM=True,
        auto_dbm=True,
        wind_type="Slow",
        w=400,
        gamma=0.7e-7,
        dt=20,
        dr0=1,
        dv0=100,
        dw=None,
        dgamma=0.1e-7,
    )
    A = downloader.P_DBM_run()
    ic(A)

    dbm.show_plots(A["Heliosphere"])
    dbm.show_plots(A["RVT_plot"])
