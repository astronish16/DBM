"""
1D_calculation: import function to perform DBM simulation in 1D
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


class DBM1D:
    def __init__(
        self,
        time_utc,
        r0,
        v0,
        target_name,
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
    ):
        self.time_utc = time_utc
        self.T0 = time_utc.timestamp()
        self.r0 = ((r0 * u.R_sun).to(u.km)).value
        self.v0 = v0
        self.target_name = target_name
        _, self.r_target = dbm.position(target_name, time_utc)
        self.r1 = ((self.r_target * u.au).to(u.km)).value
        ic(self.r1)

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

            T_array, V_array, V0_array = dbm.PDBM(
                self.r0,
                self.dr0,
                self.r1,
                self.v0,
                self.dv0,
                gamma_array,
                w_array,
                self.dt0,
                self.N,
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
            ic(t_arrival_UTC)

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
                "Target": self.target_name,
            }

        elif self.auto_dbm == True and self.P_DBM == False:
            w, gamma = dbm.auto_w_gamma_func(
                PDBM=self.P_DBM, wind_type=self.wind_type, N=1000
            )

            T1, V1 = dbm.DBM(self.r0, self.r1, self.v0, gamma, w)
            t_arrival = self.T0 + (T1 * 3600)
            t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime("%Y-%m-%d %H:%M")
            tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            ic(t_arrival_UTC)

            # Getting Plots
            rvt_plot = dbm.DBM_RVT_plot(
                self.time_utc, T1, self.r0, self.v0, gamma, w, self.r_target, tdate
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
                "Target": self.target_name,
            }

        elif self.auto_dbm == False and self.P_DBM == True:
            w_array = np.random.normal(self.w, self.dw / 3.0, self.N)
            gamma_array = np.clip(
                np.random.normal(
                    self.gamma * 1.0e-7, self.dgamma * 1.0e-7 / 3.0, self.N
                ),
                1.0e-15,
                3.0e-2,
            )
            T_array, V_array, V0_array = dbm.PDBM(
                self.r0,
                self.dr0,
                self.r1,
                self.v0,
                self.dv0,
                gamma_array,
                w_array,
                self.dt0,
                self.N,
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
            ic(t_arrival_UTC)

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
                "Target": self.target_name,
            }

        elif self.auto_dbm == False and self.P_DBM == False:
            w = self.w
            gamma = self.gamma

            T1, V1 = dbm.DBM(self.r0, self.r1, self.v0, gamma, w)
            t_arrival = self.T0 + (T1 * 3600)
            t_arrival_UTC = datetime.fromtimestamp(t_arrival).strftime("%Y-%m-%d %H:%M")
            tdate = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            T_arrival_UTC = datetime.strptime(t_arrival_UTC, "%Y-%m-%d %H:%M")
            ic(t_arrival_UTC)

            # Getting Plots
            rvt_plot = dbm.DBM_RVT_plot(
                self.time_utc, T1, self.r0, self.v0, gamma, w, self.r_target, tdate
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
                "Target": self.target_name,
            }

        return results


# ▶️ Example usage:
if __name__ == "__main__":
    downloader = DBM1D(
        time_utc=datetime(2022, 12, 1, 12, 23),
        r0=20,
        v0=1000,
        target_name="Earth",
        P_DBM=False,
        auto_dbm=True,
        wind_type="Slow",
        w=400,
        gamma=0.2e-7,
        dt=20,
        dr0=1,
        dv0=100,
        dw=None,
        dgamma=None,
    )
    A = downloader.P_DBM_run()
    ic(A)
