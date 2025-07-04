"""
dbm_functions: import function to perform DBM simulation
Author: Ronish Mugatwala
E-mail: ronish.mugatwala@edu.unige.it
Github: astronish16

Updates:
19th June 2024: new project structure.
1st July 2025: Function optimazation and use 1D function in 2D as much as possible to avoid self repetation.

"""

# Numeric imports
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
import seaborn as sns
from datetime import datetime, date, timedelta
from io import BytesIO
from PIL import Image

# setup plot
# Set up plotting
plt.style.use("ggplot")
sns.set_palette("tab10")


# Function to show plots stored in memory
def show_plots(plot_image):
    """
    Function show plots saved in memory
    Args:
        plot_image (BytesIO object):

    Returns:
        None
    """
    img = Image.open(plot_image)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    return None


# Some Lists and Array for automization purposes
Inner_Planets = ["Mercury", "Venus", "Earth", "Mars"]
Outer_Planets = ["Jupiter", "Saturn", "Uranus", "Neptune"]
Space_Crafts = [
    "Messenger",
    "VEX",
    "PSP",
    "SolO",
    "BepiCol",
    "Spitzer",
    "Wind",
    "ST-A",
    "ST-B",
    "Kepler",
    "Ulysses",
    "MSL",
    "Maven",
    "Juno",
]


objects_list = Inner_Planets + Outer_Planets + Space_Crafts


# Function required to find DBM solution
def func(t, r0, r1, v0, gamma, w):
    """_summary_

    Args:
        t (_float_): time
        r0 (_float_): initial postion of CME (km)
        r1 (_float_): target position (km)
        v0 (_float_): speed of CME at r0 (km/s)
        gamma (_float_): drag parameter (km-1) (typically: 0.2e-7)
        w (_float_): solar wind speed (km/s)

    Returns:
        _type_: _description_
    """
    if v0 >= w:
        gamma = gamma
    else:
        # only possible contion is v0<w.
        gamma = -1 * gamma

    p1 = 1 + gamma * (v0 - w) * t
    y = -r1 + r0 + w * t + (np.log(p1) / gamma)
    y1 = w + ((v0 - w) / p1)

    return y, y1


# Wrapper function for y and y1.
"""
It is necessary for the DBM function.
If other solution is there then it need to be find.
calleable function can enhance the performance
"""


def func_y(t, r0, r1, v0, gamma, w):
    y, _ = func(t, r0, r1, v0, gamma, w)
    return y


def func_y1(t, r0, r1, v0, gamma, w):
    _, y1 = func(t, r0, r1, v0, gamma, w)
    return y1


# Wrapper function to get W and gamma.
def auto_w_gamma_func(PDBM, wind_type, N):
    """
    This function generates the value of solar wind speed when auto_dbm is True.
    PDF used in this function is from Mugatwala_et_al_2024.
    Args:
        N (int): Number of ensemble
        PDBM (bool): Choice for PDBM. [True, False]
        wind_type (str): Choice for solar wind type. ["Slow","Fast"]

    Returns:
        w_array, gamma_array if PDBM is set to True otherwise provides median value for w and gamma.
    """
    if wind_type == "Slow":
        w_array = np.clip(sts.norm.rvs(370.530847, 88.585045 / 3.0, size=N), 1, 1000)
    else:
        w_array = np.clip(sts.norm.rvs(579.057905, 67.870776 / 3.0, size=N), 1, 1000)

    gamma_array = np.clip(
        sts.lognorm.rvs(
            0.6518007540612114 / 2.0,
            -2.2727287735377082e-08,
            9.425812152200486e-08,
            size=N,
        ),
        1.0e-09,
        3.0e-7,
    )
    ic(
        np.max(gamma_array),
        np.mean(gamma_array),
        np.median(gamma_array),
        np.std(gamma_array),
    )

    if PDBM == True:
        return w_array, gamma_array
    else:
        w_median = np.nanmedian(w_array)
        gamma_median = np.nanmedian(gamma_array)
        return w_median, gamma_median


# Function to plot RVT plot for DBM


def RV(t, r0, v0, gamma, w):
    """
    Calculte the distance and speed under DBM approximation.
    Args:
        t (float): time at which r and v are supposed to calculate.
        r0 (float): initial position (km)
        v0 (float): speed at r0 (km/s)
        gamma (float): drag parameter (km-1) [~0.2e-7]
        w (float): solar wind speed (km/s) [~400]

    Returns:
        r,v (float,float): distance and speed at time t (km.km/s)
    """
    if v0 >= w:
        gamma = gamma
    else:
        # only possible contion is v0<w.
        gamma = -1 * gamma

    p1 = 1 + gamma * (v0 - w) * t
    r = r0 + w * t + (np.log(p1) / gamma)
    v = w + ((v0 - w) / p1)

    return r, v


def DBM_RVT_plot(time_utc, TT, r0, v0, gamma, w, r_target, tdate):
    """
    Make distance-speed-time (R-V-T) plot for a point of interest on CME leading edge.
    Units on plot canvas: R [Solar radii], V[km/s], T[date]
    Args:
        time_utc (datetime object): time when CME is at r0
        TT (float): Transit time [hrs]
        r0 (float): initial position of CME [km]
        v0 (float): speed at r0 [km/s]
        gamma (float):
        w (float): _description_
        r_target (float): target distanec [AU]
        tdate (datetime object): CME arrival time

    Returns:
        RVT plot as BytesIO object
    """
    dt = 3600  # unit is second
    t_ary = np.arange(0, TT * 1.1 * 3600, 80)
    Time = [time_utc + timedelta(seconds=i) for i in t_ary]

    R, V = RV(t_ary, r0, v0, gamma, w)
    R = (R * u.km).to(u.R_sun).value
    r_target = (r_target * u.au).to(u.R_sun).value

    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.grid()
    color = "tab:red"
    ax1.set_xlabel("time (UTC date hour)", fontsize=17)
    ax1.set_ylabel("R (solar radius)", color=color, fontsize=17)
    ax1.plot(Time, R, color=color, label="Distance")
    ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
    ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
    ax1.tick_params(axis="y", labelcolor=color, labelsize=17)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:cyan"
    # we already handled the x-label with ax1
    ax2.set_ylabel("V (km/s)", color=color, fontsize=17)
    ax2.plot(Time, V, color=color, label="Speed")
    ax2.tick_params(axis="y", labelcolor=color, labelsize=17)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    lines = lines_1 + lines_2
    labels = labels_1 + labels_2

    ax1.legend(lines, labels, fontsize=17, loc=3)
    plt.grid(True)

    # Save the plot to an in-memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)  # Move to the start of the buffer
    plt.close()  # Close the plot to free resources

    return buffer


def PDBM_RVT_plot(
    time_utc, TT_array, r0, v0_array, gamma_array, w_array, r_target, tdate
):
    """
    Same purpose as @DBM_RVT_plot. Few inputs are changed to accomodate PDBM output, while rest remains the same.
    Args:
        TT_array (np.array): Transit time array
        v0_array (np.array): Initial speed array
        gamma_array (np.array):
        w_array (np.array):

    """
    TT = np.nanmedian(TT_array)
    t_ary = np.arange(0, TT * 1.1 * 3600, 80)
    Time = [time_utc + timedelta(seconds=i) for i in t_ary]

    i_array = np.arange(0, len(v0_array), 1)
    R_matrix = np.zeros((len(v0_array), len(t_ary)))
    V_matrix = np.zeros((len(v0_array), len(t_ary)))

    for v0, w, g, i in zip(v0_array, w_array, gamma_array, i_array):
        y, y1 = RV(t_ary, r0, v0, g, w)
        y = (y * u.km).to(u.R_sun).value
        R_matrix[i] = y
        V_matrix[i] = y1

    r_target = (r_target * u.au).to(u.R_sun).value
    R_median = np.nanmedian(R_matrix, axis=0)
    V_median = np.nanmedian(V_matrix, axis=0)
    R_max = np.max(R_matrix, axis=0)
    R_min = np.min(R_matrix, axis=0)
    V_max = np.max(V_matrix, axis=0)
    V_min = np.min(V_matrix, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.grid()
    color = "tab:red"
    ax1.set_xlabel("time (UTC date hour)", fontsize=17)
    ax1.set_ylabel("R (solar radius)", color=color, fontsize=17)
    ax1.plot(Time, R_median, color=color, label="Distance")
    ax1.fill_between(Time, R_max, R_min, alpha=0.25, linewidth=0, color=color)
    ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
    ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = "tab:cyan"
    # we already handled the x-label with ax1
    ax2.set_ylabel("V (km/s)", color=color, fontsize=17)
    ax2.plot(Time, V_median, color=color, label="Speed")
    ax2.fill_between(Time, V_max, V_min, alpha=0.25, linewidth=0, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    lines = lines_1 + lines_2
    labels = labels_1 + labels_2

    ax1.legend(lines, labels, fontsize=17, loc=3)
    plt.grid(True)

    # Save the plot to an in-memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)  # Move to the start of the buffer
    plt.close()  # Close the plot to free resources

    return buffer


def TT_plot(T):
    """
    Function to plot transit time distribution when PDBM calculations are performed.

    Args:
        T (np.array): Transit time array

    Returns:
        ByteIO object
    """
    mean = np.nanmean(T)
    median = np.nanmedian(T)
    plt.hist(T, bins=50, density=True)
    plt.xlim(np.nanmin(T), np.nanmax(T))
    plt.axvline(mean, color="red", label=f"Mean: {mean:.2f} hr ")
    plt.axvline(median, color="black", label=f"Median: {median:.2f} hr ")
    plt.title("Transit Time Distribution")
    plt.xlabel("Transit Time (hrs)")
    plt.xlim(0, np.nanmax(T))
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save the plot to an in-memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)  # Move to the start of the buffer
    plt.close()  # Close the plot to free resources

    return buffer


def V_plot(V):
    """
    Function to plot transit speed distribution when PDBM calculations are performed.

    Args:
        V (np.array): Transit speed array

    Returns:
        ByteIO object
    """
    mean = np.nanmean(V)
    median = np.nanmedian(V)
    plt.hist(V, bins=50, density=True)
    plt.xlim(np.nanmin(V), np.nanmax(V))
    plt.axvline(mean, color="red", label=f"Mean: {mean:.2f} km/s ")
    plt.axvline(median, color="black", label=f"Median: {median:.2f} km/s ")
    plt.title("Arrival Speed Distribution")
    plt.xlabel("Arrival Speed (km/s)")
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Save the plot to an in-memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)  # Move to the start of the buffer
    plt.close()  # Close the plot to free resources

    return buffer


"""
-------------------------------------------------------------------------------  
 __   _______       ___.______   ___   _______  .______   .___  ___. 
/_ | |       \     /  /|   _  \  \  \ |       \ |   _  \  |   \/   | 
 | | |  .--.  |   |  | |  |_)  |  |  ||  .--.  ||  |_)  | |  \  /  | 
 | | |  |  |  |   |  | |   ___/   |  ||  |  |  ||   _  <  |  |\/|  | 
 | | |  '--'  |   |  | |  |       |  ||  '--'  ||  |_)  | |  |  |  | 
 |_| |_______/    |  | | _|       |  ||_______/ |______/  |__|  |__| 
                   \__\          /__/                                
-------------------------------------------------------------------------------
"""

# Function to find DBM solution.
"""
This function provides transist time and impact speed
of CME under DBM approximation
"""


def DBM(r0, r1, v0, gamma, w):
    """
    This is main function to determine the transit time and speed of CME under DBM approximation.

    Args:
        r0 (float): initial position (km)
        r1 (float): target distance (km)
        v0 (float): speed at r0 (km/s)
        gamma (float): drag parameter (km-1) [~0.2e-7]
        w (float): solar wind speed (km/s) [~400]

    Returns:
        t1 (float) : transit time of CME (hrs)
        v1 (float) : transit speed of CME (km/s)
    """
    t1 = newton(
        func=func_y,
        fprime=func_y1,
        x0=30 * 3600,
        args=[r0, r1, v0, gamma, w],
        disp=False,
        maxiter=30,
    )

    dv = v0 - w
    p1 = 1 + (gamma * np.abs(dv) * t1)
    v1 = w + (dv / p1)

    # Transit time is in hours and impact speed is in km/s.
    return t1 / 3600.0, v1


def PDBM(r0, dr0, r1, v0, dv0, gamma_array, wind_array, dt0, N):
    """
    This function perform PDBM calculations.
    Args:
        r0 (float): initial position (km)
        dr0 (float): uncertainity in r0 (km)
        r1 (float): target distance (km)
        v0 (float): speed at r0 (km/s)
        dv0 (floar): uncertainity in v0 (km/s)
        gamma_array (list / numpy array): collection of gamma values either from PDF or manual
        wind_array (list / numpy array ): collection of w values
        dt0 (float): uncertainity in t0 (s)
        N (int): number of ensembels

    Returns:
        TT_array (numpy array): collection of transit time (values are in hours)
        V_array (numpy array): collection of transit speed (values are in km/s)
        V0_array (numpy array): collection of v0 used in calculation.
    """
    r0_array = np.random.normal(r0, dr0 / 3.0, N)
    v0_array = np.random.normal(v0, dv0 / 3.0, N)
    r1_array = np.random.normal(
        r1, 0.05 * r1 / 3.0, N
    )  # including 5% error in the target distace
    t0_array = np.random.normal(0, dt0 / (60.0 * 3), N)

    # Arrays to store Output
    TT_array = np.zeros_like(r0_array)
    V_array = np.zeros_like(r0_array)

    for i in range(0, N):
        TT_array[i], V_array[i] = DBM(
            r0_array[i], r1_array[i], v0_array[i], gamma_array[i], wind_array[i]
        )
        TT_array[i] = TT_array[i] + t0_array[i]

    return TT_array, V_array, v0_array


# Function for 2D DBM
"""
.------------------------------------------------------------------.
| ,---. ,------.       ,-.,------.,-. ,------.  ,-----. ,--.   ,--.|
|'.-.  \|  .-.  \     / .'|  .--. '. \|  .-.  \ |  |) /_|   `.'   ||
| .-' .'|  |  \  :   |  | |  '--' ||  |  |  \  :|  .-.  \  |'.'|  ||
|/   '-.|  '--'  /   |  | |  | --' |  |  '--'  /|  '--' /  |   |  ||
|'-----'`-------'     \ '.`--'    .' /`-------' `------'`--'   `--'|
|                      `-'        `-'                              |
'------------------------------------------------------------------'
"""

"""
while moving to 2D version of model,one has to consider two important variable.
(1) CME Propagation Direction
(2) Target longitude.
The absolute difference of these two quantity is called alpha.
Also, CME propagation direction has been measured with respect to the Earth in cone geometry.
Therefore, we also need to consider coordinate system.
"""

# Correction in central meridian as per Heliocentric Ecliptic coordinate system.
"""
We are doing this because we are using JPL Horiozn for ephemeris to determine target information.
In JPL Horizon Heliocentric ecliptic coordinate system is used so position of Earth is not Fixed.
While in cone model Heliocentric Stonyhurst coordinate system is used where Sun-Earth line is always
correspond to the 0$^o$ longitude.
"""


def Phi_Correction(phi_cme, time_utc):
    """
    Args:
        phi_cme (float): central meridian of CME [deg]
        time_utc (datetime object): time correspond to r0

    Return:
        phi_corrected (float): corrected central meridian of CME
    """
    earth = ephem.Sun()
    earth.compute(time_utc)
    phi_corrected = np.rad2deg(np.deg2rad(phi_cme) + earth.hlon)
    return phi_corrected


"""
For 2D cone, There are 3 possiblities (see Schwenn et al, 2005)
(1) ICME leading edge is concentric arc with solar surface. keyword: ["CC", "Concentric Cone"]
    The application of this geometry is same as 1D DBM.
(2) ICME leading edge is semi circle. keyword: [""IC", "Ice-Cream Cone"]
    This geometry looks like a ice cream cone
(3) ICME leading edge is circular arc and tangentially connect to the ICME legs. keyword: ["TC", "Tangential Cone"]
    Application of this geometry is bit difficult.
"""

"""
When we consider a geometry, possiblity of two different type of evolution is arise.
(1) Self Similar Expansion: CME maintain it's shape during propagation. keyword: ["SSE", "Self-Similar Expansion"]
(2) Flattening Cone Evolution: Each and every point on CME edge follows DBM. keyword: ["FCE", "Flattening Cone Evolution"]
For more detailed informatio: Check the documantation.
"""

# Function to calculate speed and distance at alpha angle


# ICE Cream Cone only
def IC_RV_alpha(omega, alpha, r0, v0):
    """
    Provides ditance and speed of CME point located at angle alpha under Ice-Cream cone approximation.

    Args:
        omega (float): half angular windth of cone [deg]
        alpha (float): interested angle on CME leading edge
        r0 (float): CME apex position
        v0 (float): CME apex speed

    Returns:
        r01,v01 (float): distance and speed of CME point at angle alpha.
    """
    omega = np.deg2rad(omega)
    alpha = np.deg2rad(alpha)
    r01 = (
        r0
        * (np.cos(alpha) + ((np.tan(omega)) ** 2 - (np.sin(alpha)) ** 2) ** 0.5)
        / (1 + np.tan(omega))
    )
    v01 = (
        v0
        * (np.cos(alpha) + ((np.tan(omega)) ** 2 - (np.sin(alpha)) ** 2) ** 0.5)
        / (1 + np.tan(omega))
    )
    return r01, v01


def IC_R_alpha_inv(omega, alpha, r1):
    """
    This function determines the CME apex distance from distance of alpha element under Ice-Cream Cone approximation.
    Args:
        omega (float): half angular width of CME cone
        alpha (float): angular seperation of considered element from the apex
        r1 (float): distance of considered element on CME leading edge.

    Returns:
        r1_apex (float): distance of CME apex point
    """
    omega = np.deg2rad(omega)
    alpha = np.deg2rad(alpha)
    r1_apex = (
        r1
        * (1 + np.tan(omega))
        / ((np.cos(alpha)) + (((np.tan(omega)) ** 2.0 - (np.sin(alpha)) ** 2.0) ** 0.5))
    )
    return r1_apex


def TC_RV_alpha(omega, alpha, r0, v0):
    """
    same as @IC_RV_alpha but consider tangential cone.
    """
    omega = np.deg2rad(omega)
    alpha = np.deg2rad(alpha)
    r01 = (
        r0
        * (np.cos(alpha) + ((np.sin(omega)) ** 2 - (np.sin(alpha)) ** 2) ** 0.5)
        / (1 + np.sin(omega))
    )
    v01 = (
        v0
        * (np.cos(alpha) + ((np.sin(omega)) ** 2 - (np.sin(alpha)) ** 2) ** 0.5)
        / (1 + np.sin(omega))
    )
    return r01, v01


def TC_R_alpha_inv(omega, alpha, r1):
    """
    same as @IC_R_alpha_inv but consider tangential cone.
    """
    omega = np.deg2rad(omega)
    alpha = np.deg2rad(alpha)
    r1_apex = (
        r1
        * (1 + np.sin(omega))
        / ((np.cos(alpha)) + (((np.sin(omega)) ** 2.0 - (np.sin(alpha)) ** 2.0) ** 0.5))
    )
    return r1_apex


def DBM_2D_RVT_plot(
    time_utc,
    TT,
    r0,
    v0,
    gamma,
    w,
    r_target,
    tdate,
    omega,
    phi_cme,
    phi_target,
    cone_geometry,
    kinematic,
):
    """
    Same functionality as @DBM_RVT_plot but for 2D.
    Majority of inputs and output are same. Few are added for 2D consideration.
    Args:
        omega (float): half angular width of CME [deg]
        phi_cme (float): central meridian of CME [deg]
        phi_target (float): logitude of target [deg]
        cone_geometry (str): type of cone
        kinematic (str): kinematic approach for CME propagation.

    Raises:
        ValueError: unknown cone geometry during SSE
        ValueError: unknown cone geometry during FCE
        ValueError: unknown kinematic approach

    Optimization over functions:
        DBM_2D_RVT_IC_SSE_plot,
        DBM_2D_RVT_TC_SSE_plot,
        DBM_2D_RVT_IC_FCE_plot,
        DBM_2D_RVT_TC_FCE_plot


    """
    alpha = np.abs(phi_target - phi_cme)
    t_ary = np.arange(0, TT * 1.1 * 3600, 80)
    Time = [time_utc + timedelta(seconds=i) for i in t_ary]
    r_target = (r_target * u.au).to(u.R_sun).value

    # This is main part to define which kind of initial transformation has to be used.
    if kinematic in ["SSE", "Self-Similar Expansion"]:
        R, V = RV(t_ary, r0, v0, gamma, w)
        R = (R * u.km).to(u.R_sun).value

        if cone_geometry in ["IC", "Ice-Cream Cone"]:
            R_ary, V_ary = IC_RV_alpha(omega, alpha, R, V)
        elif cone_geometry in ["TC", "Tangential Cone"]:
            R_ary, V_ary = TC_RV_alpha(omega, alpha, R, V)
        elif cone_geometry in ["CC", "Concentric Cone"]:
            R_ary = R.copy()
            V_ary = V.copy()
            ic(R_ary, V_ary)

        else:
            raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")

    elif kinematic in ["FCE", "Flattening Cone Evolution"]:
        if cone_geometry in ["IC", "Ice-Cream Cone"]:
            R0_a, V0_a = IC_RV_alpha(omega, alpha, r0, v0)
        elif cone_geometry in ["TC", "Tangential Cone"]:
            R0_a, V0_a = TC_RV_alpha(omega, alpha, r0, v0)
        elif cone_geometry in ["CC", "Concentric Cone"]:
            R0_a, V0_a = r0, v0
        else:
            raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")

        R, V = RV(t_ary, R0_a, V0_a, gamma, w)
        R_ary, V_ary = (R * u.km).to(u.R_sun).value, V

    else:
        raise ValueError(f"Unknown Kinematic Approach: {kinematic}")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.grid()
    color_dist = "tab:red"
    color_speed = "tab:cyan"

    ax1.set_xlabel("time (UTC date hour)", fontsize=17)
    ax1.set_ylabel("R (solar radius)", color=color_dist, fontsize=17)
    ax1.plot(Time, R_ary, color=color_dist, label="Distance")
    ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
    ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
    ax1.tick_params(axis="y", labelcolor=color_dist)

    ax2 = ax1.twinx()
    ax2.set_ylabel("V (km/s)", color=color_speed, fontsize=17)
    ax2.plot(Time, V_ary, color=color_speed, label="Speed")
    ax2.tick_params(axis="y", labelcolor=color_speed)

    fig.tight_layout()

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        fontsize=17,
    )
    plt.grid(True)
    plt.title("2D RVT Plot", fontsize=17)
    plt.tight_layout()

    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    plt.close()

    return buffer


def PDBM_2D_RVT_plot(
    time_utc,
    TT_array,
    r0,
    v0_array,
    gamma_array,
    w_array,
    r_target,
    tdate,
    omega_array,
    phi_cme_array,
    phi_target,
    cone_geometry,
    kinematic,
):
    """
    same application DBM_2D_RVT_plot but for probabilistic case.
    Inputs are changed to accomodate the PDBM results.

    Args:
        TT_array (np.array): transit time array
        v0_array (np.array): initial speed array
        gamma_array (np.array):
        w_array (np.array):
        omega_array (np.array): half angular width array
        phi_cme_array (np.array): CME central meridian array

    """
    TT = np.nanmedian(TT_array)
    t_ary = np.arange(0, TT * 1.1 * 3600, 80)
    Time = [time_utc + timedelta(seconds=i) for i in t_ary]
    r_target = (r_target * u.au).to(u.R_sun).value
    alpha_array = np.abs(phi_cme_array - phi_target)
    # ensure each alpha[i] ≤ omega_array[i]
    # alpha_array = np.minimum(alpha_array, omega_array)

    i_array = np.arange(0, len(v0_array), 1)
    R_matrix = np.zeros((len(v0_array), len(t_ary)))
    V_matrix = np.zeros((len(v0_array), len(t_ary)))

    # This is main part to define which kind of initial transformation has to be used.
    if kinematic in ["SSE", "Self-Similar Expansion"]:
        for v0, w, g, omeg, alph, i in zip(
            v0_array, w_array, gamma_array, omega_array, alpha_array, i_array
        ):
            R, V = RV(t_ary, r0, v0, g, w)
            R = (R * u.km).to(u.R_sun).value

            if cone_geometry in ["IC", "Ice-Cream Cone"]:
                R_ary, V_ary = IC_RV_alpha(omeg, alph, R, V)
                # ic(V_ary)
            elif cone_geometry in ["TC", "Tangential Cone"]:
                R_ary, V_ary = TC_RV_alpha(omeg, alph, R, V)
            elif cone_geometry in ["CC", "Concentric Cone"]:
                R_ary = R.copy()
                V_ary = V.copy()
                # ic(V_ary)
            else:
                raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")

            R_matrix[i], V_matrix[i] = R_ary, V_ary

    elif kinematic in ["FCE", "Flattening Cone Evolution"]:
        for v0, w, g, omeg, alph, i in zip(
            v0_array, w_array, gamma_array, omega_array, alpha_array, i_array
        ):
            if cone_geometry in ["IC", "Ice-Cream Cone"]:
                R0_a, V0_a = IC_RV_alpha(omeg, alph, r0, v0)
            elif cone_geometry in ["TC", "Tangential Cone"]:
                R0_a, V0_a = TC_RV_alpha(omeg, alph, r0, v0)
            elif cone_geometry in ["CC", "Concentric Cone"]:
                R0_a, V0_a = r0, v0
            else:
                raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")

            R, V = RV(t_ary, R0_a, V0_a, g, w)
            R_ary, V_ary = (R * u.km).to(u.R_sun).value, V
            R_matrix[i], V_matrix[i] = R_ary, V_ary

    else:
        raise ValueError(f"Unknown Kinematic Approach: {kinematic}")

    R_median = np.nanmedian(R_matrix, axis=0)
    V_median = np.nanmedian(V_matrix, axis=0)
    R_max = np.max(R_matrix, axis=0)
    R_min = np.min(R_matrix, axis=0)
    V_max = np.max(V_matrix, axis=0)
    V_min = np.min(V_matrix, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.grid()
    color_dist = "tab:red"
    color_speed = "tab:cyan"

    ax1.set_xlabel("time (UTC date hour)", fontsize=17)
    ax1.set_ylabel("R (solar radius)", color=color_dist, fontsize=17)
    ax1.plot(Time, R_median, color=color_dist, label="Distance")
    ax1.fill_between(Time, R_max, R_min, alpha=0.25, linewidth=0, color=color_dist)
    ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
    ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
    ax1.tick_params(axis="y", labelcolor=color_dist)

    ax2 = ax1.twinx()
    ax2.set_ylabel("V (km/s)", color=color_speed, fontsize=17)
    ax2.plot(Time, V_median, color=color_speed, label="Speed")
    ax2.fill_between(Time, V_max, V_min, alpha=0.25, linewidth=0, color=color_speed)
    ax2.tick_params(axis="y", labelcolor=color_speed)

    fig.tight_layout()

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        fontsize=17,
    )
    plt.grid(True)
    plt.title("2D RVT Plot", fontsize=17)
    plt.tight_layout()

    # Save to buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    plt.close()

    return buffer


# def DBM_2D_RVT_IC_SSE_plot(
#     time_utc, TT, r0, v0, gamma, w, r_target, tdate, omega, alpha
# ):
#     t_ary = np.arange(0, TT * 1.1 * 3600, 80)
#     Time = [time_utc + timedelta(seconds=i) for i in t_ary]

#     R, V = RV(t_ary, r0, v0, gamma, w)
#     R = (R * u.km).to(u.R_sun).value
#     r_target = (r_target * u.au).to(u.R_sun).value

#     R_ary, V_ary = IC_RV_alpha(omega, alpha, R, V)

#     plt.style.use("seaborn-v0_8-darkgrid")
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     plt.grid()
#     color = "tab:red"
#     ax1.set_xlabel("time (UTC date hour)", fontsize=17)
#     ax1.set_ylabel("R (solar radius)", color=color, fontsize=17)
#     ax1.plot(Time, R_ary, color=color, label="Distance")
#     ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
#     ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
#     ax1.tick_params(axis="y", labelcolor=color)

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#     color = "tab:cyan"
#     # we already handled the x-label with ax1
#     ax2.set_ylabel("V (km/s)", color=color, fontsize=17)
#     ax2.plot(Time, V_ary, color=color, label="Speed")
#     ax2.tick_params(axis="y", labelcolor=color)

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped

#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()

#     lines = lines_1 + lines_2
#     labels = labels_1 + labels_2

#     ax1.legend(lines, labels, fontsize=17, loc=3)
#     plt.grid(True)

#     # Save the plot to an in-memory buffer
#     buffer = BytesIO()
#     plt.savefig(buffer, format="png")
#     buffer.seek(0)  # Move to the start of the buffer
#     plt.close()  # Close the plot to free resources

#     return buffer


# def DBM_2D_RVT_IC_FCE_plot(
#     time_utc, TT, r0, v0, gamma, w, r_target, tdate, omega, alpha
# ):
#     t_ary = np.arange(0, TT * 1.1 * 3600, 80)
#     Time = [time_utc + timedelta(seconds=i) for i in t_ary]

#     R0_a, V0_a = IC_RV_alpha(omega, alpha, r0, v0)

#     R, V = RV(t_ary, R0_a, V0_a, gamma, w)
#     R = (R * u.km).to(u.R_sun).value
#     r_target = (r_target * u.au).to(u.R_sun).value

#     plt.style.use("seaborn-v0_8-darkgrid")
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     plt.grid()
#     color = "tab:red"
#     ax1.set_xlabel("time (UTC date hour)", fontsize=17)
#     ax1.set_ylabel("R (solar radius)", color=color, fontsize=17)
#     ax1.plot(Time, R, color=color, label="Distance")
#     ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
#     ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
#     ax1.tick_params(axis="y", labelcolor=color)

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#     color = "tab:cyan"
#     # we already handled the x-label with ax1
#     ax2.set_ylabel("V (km/s)", color=color, fontsize=17)
#     ax2.plot(Time, V, color=color, label="Speed")
#     ax2.tick_params(axis="y", labelcolor=color)

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped

#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()

#     lines = lines_1 + lines_2
#     labels = labels_1 + labels_2

#     ax1.legend(lines, labels, fontsize=17, loc=3)
#     plt.grid(True)

#     # Save the plot to an in-memory buffer
#     buffer = BytesIO()
#     plt.savefig(buffer, format="png")
#     buffer.seek(0)  # Move to the start of the buffer
#     plt.close()  # Close the plot to free resources

#     return buffer


# def DBM_2D_RVT_TC_SSE_plot(
#     time_utc, TT, r0, v0, gamma, w, r_target, tdate, omega, alpha
# ):
#     t_ary = np.arange(0, TT * 1.1 * 3600, 80)
#     Time = [time_utc + timedelta(seconds=i) for i in t_ary]

#     R, V = RV(t_ary, r0, v0, gamma, w)
#     R = (R * u.km).to(u.R_sun).value
#     r_target = (r_target * u.au).to(u.R_sun).value

#     R_ary, V_ary = TC_RV_alpha(omega, alpha, R, V)

#     plt.style.use("seaborn-v0_8-darkgrid")
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     plt.grid()
#     color = "tab:red"
#     ax1.set_xlabel("time (UTC date hour)", fontsize=17)
#     ax1.set_ylabel("R (solar radius)", color=color, fontsize=17)
#     ax1.plot(Time, R_ary, color=color, label="Distance")
#     ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
#     ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
#     ax1.tick_params(axis="y", labelcolor=color)

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#     color = "tab:cyan"
#     # we already handled the x-label with ax1
#     ax2.set_ylabel("V (km/s)", color=color, fontsize=17)
#     ax2.plot(Time, V_ary, color=color, label="Speed")
#     ax2.tick_params(axis="y", labelcolor=color)

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped

#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()

#     lines = lines_1 + lines_2
#     labels = labels_1 + labels_2

#     ax1.legend(lines, labels, fontsize=17, loc=3)
#     plt.grid(True)

#     # Save the plot to an in-memory buffer
#     buffer = BytesIO()
#     plt.savefig(buffer, format="png")
#     buffer.seek(0)  # Move to the start of the buffer
#     plt.close()  # Close the plot to free resources

#     return buffer


# def DBM_2D_RVT_TC_FCE_plot(
#     time_utc, TT, r0, v0, gamma, w, r_target, tdate, omega, alpha
# ):
#     t_ary = np.arange(0, TT * 1.1 * 3600, 80)
#     Time = [time_utc + timedelta(seconds=i) for i in t_ary]

#     R0_a, V0_a = TC_RV_alpha(omega, alpha, r0, v0)

#     R, V = RV(t_ary, R0_a, V0_a, gamma, w)
#     R = (R * u.km).to(u.R_sun).value
#     r_target = (r_target * u.au).to(u.R_sun).value

#     plt.style.use("seaborn-v0_8-darkgrid")
#     fig, ax1 = plt.subplots(figsize=(10, 6))
#     plt.grid()
#     color = "tab:red"
#     ax1.set_xlabel("time (UTC date hour)", fontsize=17)
#     ax1.set_ylabel("R (solar radius)", color=color, fontsize=17)
#     ax1.plot(Time, R, color=color, label="Distance")
#     ax1.axvline(tdate, linestyle="--", color="black", label="Arrival Time")
#     ax1.axhline(r_target, label=f"R_target = {r_target:.2f}")
#     ax1.tick_params(axis="y", labelcolor=color)

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#     color = "tab:cyan"
#     # we already handled the x-label with ax1
#     ax2.set_ylabel("V (km/s)", color=color, fontsize=17)
#     ax2.plot(Time, V, color=color, label="Speed")
#     ax2.tick_params(axis="y", labelcolor=color)

#     fig.tight_layout()  # otherwise the right y-label is slightly clipped

#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()

#     lines = lines_1 + lines_2
#     labels = labels_1 + labels_2

#     ax1.legend(lines, labels, fontsize=17, loc=3)
#     plt.grid(True)

#     # Save the plot to an in-memory buffer
#     buffer = BytesIO()
#     plt.savefig(buffer, format="png")
#     buffer.seek(0)  # Move to the start of the buffer
#     plt.close()  # Close the plot to free resources

#     return buffer


"""
in self similar expansion we need to find the position of apex first.
Then we need to solve the boundary condition accordingly,
as target distance could be anywhere along the ICME leading edge.
With this corrected boundary condition we get a Travel time and arrival speed of CME apex.
Travel time doesn't change with angle correction.
But arrival spped do chnage.
So, we need to find the cosine component of CME apex speed which arrival speed of CME at target.
"""


# def DBM_IC_SSE(r0, r1, v0, gamma, w, omega, alpha):
#     R1_apex = IC_R_alpha_inv(omega, alpha, r1)
#     TT, V1_apex = DBM(r0, R1_apex, v0, gamma, w)
#     _, V1 = IC_RV_alpha(omega, alpha, R1_apex, V1_apex)
#     return TT, V1


# def DBM_TC_SSE(r0, r1, v0, gamma, w, omega, alpha):
#     R1_apex = TC_R_alpha_inv(omega, alpha, r1)
#     TT, V1_apex = DBM(r0, R1_apex, v0, gamma, w)
#     _, V1 = TC_RV_alpha(omega, alpha, R1_apex, V1_apex)
#     return TT, V1


# def DBM_IC_FCE(r0, r1, v0, gamma, w, omega, alpha):
#     r0_a, v0_a = IC_RV_alpha(omega, alpha, r0, v0)
#     TT, V1 = DBM(r0_a, r1, v0_a, gamma, w)
#     return TT, V1


# def DBM_TC_FCE(r0, r1, v0, gamma, w, omega, alpha):
#     r0_a, v0_a = TC_RV_alpha(omega, alpha, r0, v0)
#     TT, V1 = DBM(r0_a, r1, v0_a, gamma, w)
#     return TT, V1


def DBM_2D(r0, r1, v0, gamma, w, omega, phi_cme, phi_target, cone_geometry, kinematic):
    """
    Main function that perfrom DBM calculation in 2D.
    Majority of inputs and outputs are same @DBM.
    Some new input are for considering mentioned posibilities for 2D.

    Args:
        omega (float): half angular width of CME [deg]
        phi_cme (float): central meridian of CME [deg]
        phi_target (float): logitude of target [deg]
        cone_geometry (str): type of cone
        kinematic (str): kinematic approach for CME propagation.


    Raises:
        ValueError: unknown cone geometry during SSE
        ValueError: unknown cone geometry during FCE
        ValueError: unknown kinematic approach

    Optimization over functions:
        DBM_IC_SSE,
        DBM_IC_FCE,
        DBM_TC_SSE,
        DBM_TC_FCE
    """
    # angle at which we have to perfrom (P-)DBM calculation
    alpha = np.abs(phi_target - phi_cme)
    if kinematic == "SSE" or "Self-Similar Expansion":
        if cone_geometry in ["IC", "Ice-Cream Cone"]:
            R1_apex = IC_R_alpha_inv(omega, alpha, r1)
            TT, V1_apex = DBM(r0, R1_apex, v0, gamma, w)
            _, V1 = IC_RV_alpha(omega, alpha, R1_apex, V1_apex)
        elif cone_geometry in ["TC", "Tangential Cone"]:
            R1_apex = TC_R_alpha_inv(omega, alpha, r1)
            TT, V1_apex = DBM(r0, R1_apex, v0, gamma, w)
            _, V1 = TC_RV_alpha(omega, alpha, R1_apex, V1_apex)
        elif cone_geometry in ["CC", "Concentric Cone"]:
            TT, V1 = DBM(r0, r1, v0, gamma, w)
        else:
            raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")

    elif kinematic in ["FCE", "Flattening Cone Evolution"]:
        if cone_geometry in ["IC", "Ice-Cream Cone"]:
            r0_a, v0_a = IC_RV_alpha(omega, alpha, r0, v0)
        elif cone_geometry in ["TC", "Tangential Cone"]:
            r0_a, v0_a = TC_RV_alpha(omega, alpha, r0, v0)
        else:
            raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")

        TT, V1 = DBM(r0_a, r1, v0_a, gamma, w)

    else:
        raise ValueError(f"Unknown Kinematic: {kinematic}")

    return TT, V1


def PDBM_2D(
    r0,
    dr0,
    r1,
    v0,
    dv0,
    gamma_array,
    wind_array,
    omega,
    domega,
    phi_cme,
    dphi_cme,
    phi_target,
    cone_geometry,
    kinematic,
    dt0,
    N,
):
    """
    Main function to perfrom PDBM calculations in 2D.

    Args:
        r0 (float):
        dr0 (float): uncertanity in r0
        r1 (float):
        dv0 (float): uncertanity in v0
        gamma_array (np.array):
        wind_array (np.array):
        omega (float):
        domega (float): uncertanity in omega
        phi_cme (float):
        dphi_cme (float): uncertanity in phi_cme
        phi_target (float):
        cone_geometry (str):
        kinematic (str):
        dt0 (float): uncertanity in t0
        N (int): number of ensemble

    Returns:
        TT_array[valid_mask] (np.array): Transit time array (only for ensembles when CME hits the target)
        V_array[valid_mask] (np.array): Transit speed array (only for ensembles when CME hits the target)
        Hits (int): Number of ensembles when CME hits a target
        Miss(int): Number of ensembles when CME misses a target
        v0_array[valid_mask] (np.array): initial spped array (only for ensembles when CME hits the target)
        omega_array[valid_mask] (np.array): Transit spped array (only for ensembles when CME hits the target)
        phi_cme_array[valid_mask] (np.array): Transit spped array (only for ensembles when CME hits the target)

    """
    r0_array = np.random.normal(r0, dr0 / 3.0, N)
    v0_array = np.random.normal(v0, dv0 / 3.0, N)
    r1_array = np.random.normal(
        r1, 0.05 * r1 / 3.0, N
    )  # including 5% error in the target distace
    t0_array = np.random.normal(0, dt0 / (60.0 * 3), N)
    if cone_geometry in ["CC", "Concentric Cone"]:
        omega_array = np.random.normal(omega, domega / 3.0, N)
    else:
        omega_array = np.clip(np.random.normal(omega, domega / 3.0, N), 0, 90)
    phi_cme_array = np.random.normal(phi_cme, dphi_cme / 3.0, N)

    # Arrays to store Output
    TT_array = np.full(N, np.nan)
    V_array = np.full(N, np.nan)
    TT_miss_array = np.full(N, np.nan)
    V_miss_array = np.full(N, np.nan)

    Hits = 0
    Miss = 0

    for i in range(0, N):
        if (
            (phi_cme_array[i] - omega_array[i])
            <= phi_target
            <= (phi_cme_array[i] + omega_array[i])
        ):
            Hits = Hits + 1
            TT_array[i], V_array[i] = DBM_2D(
                r0_array[i],
                r1_array[i],
                v0_array[i],
                gamma_array[i],
                wind_array[i],
                omega_array[i],
                phi_cme_array[i],
                phi_target,
                cone_geometry,
                kinematic,
            )
            TT_array[i] = TT_array[i] + t0_array[i]
        else:
            Miss = Miss + 1
            TT_miss_array[i], V_miss_array[i] = DBM(
                r0_array[i],
                r1_array[i],
                v0_array[i],
                gamma_array[i],
                wind_array[i],
            )
            TT_miss_array[i] = TT_miss_array[i] + t0_array[i]

    if Hits == 0:
        valid_mask = ~np.isnan(TT_miss_array)
        return (
            TT_miss_array[valid_mask],
            V_miss_array[valid_mask],
            Hits,
            Miss,
            v0_array[valid_mask],
            omega_array[valid_mask],
            phi_cme_array[valid_mask],
        )

    else:
        valid_mask = ~np.isnan(TT_array)
        return (
            TT_array[valid_mask],
            V_array[valid_mask],
            Hits,
            Miss,
            v0_array[valid_mask],
            omega_array[valid_mask],
            phi_cme_array[valid_mask],
        )


# def DBM_Self_Similar_Expansion(r0, r1, v0, gamma, w, omega, alpha):
#     R1_apex = R_alpha_inv(omega, alpha, r1)
#     TT, V1_apex = DBM(r0, R1_apex, v0, gamma, w)
#     _, V1 = RV_alpha(omega, alpha, R1_apex, V1_apex)
#     return TT, V1


# def Forecast_SSE(r0, r1, v0, gamma, w, omega, phi_cme, phi_target):
#     if (phi_cme - omega) <= phi_target <= (phi_cme + omega):
#         print("Ohh no! CME hits the target")
#         print("Space Weather Alert")
#         alpha = np.abs(phi_cme - phi_target)
#         TT, V1 = DBM_Self_Similar_Expansion(r0, r1, v0, gamma, w, omega, alpha)
#     else:
#         print("Yay !!! CME misses the target")
#         print("Model Calculation for Research Purpose")
#         print("1D-DBM Model values")
#         TT, V1 = DBM(r0, r1, v0, gamma, w)
#     return TT, V1


# def DBM_Flattening_Cone(r0, r1, v0, gamma, w, omega, alpha):
#     r0_a, v0_a = RV_alpha(omega, alpha, r0, v0)
#     TT, V1 = DBM(r0_a, r1, v0_a, gamma, w)
#     return TT, V1


# def Forecast_FC(r0, r1, v0, gamma, w, omega, phi_cme, phi_target):
#     if (phi_cme - omega) <= phi_target <= (phi_cme + omega):
#         print("Ohh no! CME hits the target")
#         print("Space Weather Alert")
#         alpha = np.abs(phi_cme - phi_target)
#         TT, V1 = DBM_Flattening_Cone(r0, r1, v0, gamma, w, omega, alpha)
#     else:
#         print("Yay !!! CME misses the target")
#         print("Model Calculation for Research Purpose")
#         print("1D-DBM Model values")
#         TT, V1 = DBM(r0, r1, v0, gamma, w)
#     return TT, V1


def horizon_id(target):
    """
    This function provides JPL Horizon Id for the target.
    This id is futhuer used to determine ephemeries of target.

    Args:
        target (str): keyword for planets/spacecraft

    Returns:
        h_id (int)
    """
    if target not in objects_list:
        print("Provided heliospheric object doesn't exist in the object list")
        print("Please check the object list to find correct object name.")
        print(f"Available Objects: {objects_list}")
        sys.exit()
    elif target == "Mercury":
        h_id = 199
    elif target == "Venus":
        h_id = 299
    elif target == "Earth":
        h_id = 399
    elif target == "Mars":
        h_id = 499
    elif target == "Jupiter":
        h_id = 599
    elif target == "Saturn":
        h_id = 699
    elif target == "Uranus":
        h_id = 799
    elif target == "Neptune":
        h_id = 899
    elif target == "Messenger":
        h_id = -236
    elif target == "VEX":
        h_id = -248
    elif target == "PSP":
        h_id = -96
    elif target == "SolO":
        h_id = -144
    elif target == "BepiCol":
        h_id = -121
    elif target == "Spitzer":
        h_id = -79
    elif target == "Wind":
        h_id = -8
    elif target == "ST-A":
        h_id = -234
    elif target == "ST-B":
        h_id = -235
    elif target == "Kepler":
        h_id = -227
    elif target == "Ulysses":
        h_id = -55
    elif target == "MSL":
        h_id = -76
    elif target == "Maven":
        h_id = -202
    elif target == "Juno":
        h_id = -61

    return str(h_id)


def position(target, date):
    """
    This function fetch the ephemeries of the target on provided date.

    Args:
        target (str): name of target
        date (datetime object):

    Returns:
        phi_obj (float): logitude of target in JPL Horizon system.
        r_obj (float): heliocentric distance of the target.
    """
    tjd = Time(date).jd
    h_id = horizon_id(target)
    try:
        ic(tjd)
        obj = Horizons(id=h_id, location="@sun", epochs=tjd)
        obj_eph = obj.ephemerides()
        obj_name = obj_eph["targetname"]
        phi_obj = obj_eph["EclLon"][0]
        r_obj = obj_eph["r"][0]
    except Exception as e:
        ic(e)
        print(f"No {target} during the provided date")
        phi_obj = np.nan
        r_obj = np.nan

    return phi_obj, r_obj


def CME_edge(tt, r0, v0, gamma, w, omega, phi_cme, cone_geometry, kinematic):
    tt = tt * 3600
    rad = np.linspace(phi_cme - omega, phi_cme + omega, num=30, endpoint=True)
    alphas = rad - phi_cme

    # This is main part to define which kind of initial transformation is used.
    if kinematic in ["SSE", "Self-Similar Expansion"]:
        R_apex, _ = RV(tt, r0, v0, gamma, w)
        # ic((R_apex * u.km).to(u.au).value)
        if cone_geometry in ["IC", "Ice-Cream Cone"]:
            R_ary, __ = IC_RV_alpha(omega, alphas, R_apex, _)
        elif cone_geometry in ["TC", "Tangential Cone"]:
            R_ary, __ = TC_RV_alpha(omega, alphas, R_apex, _)
        elif cone_geometry in ["CC", "Concentric Cone"]:
            R_ary = np.full(len(rad), R_apex)
        else:
            raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")
    elif kinematic in ["FCE", "Flattening Cone Evolution"]:
        R_ary = []
        if cone_geometry in ["IC", "Ice-Cream Cone"]:
            for a in alphas:
                R0_a, V0_a = IC_RV_alpha(omega, a, r0, v0)
                r, _ = RV(tt, R0_a, V0_a, gamma, w)
                R_ary.append(r)
        elif cone_geometry in ["TC", "Tangential Cone"]:
            for a in alphas:
                R0_a, V0_a = TC_RV_alpha(omega, a, r0, v0)
                r, _ = RV(tt, R0_a, V0_a, gamma, w)
                R_ary.append(r)
        elif cone_geometry in ["CC", "Concentric Cone"]:
            R_ary = np.full(len(rad), R_apex)

        else:
            raise ValueError(f"Unknown Cone Geometry: {cone_geometry}")

    R_ary = np.array(R_ary)
    R_edge = (R_ary * u.km).to(u.au).value
    angle_rad = np.deg2rad(rad)
    return angle_rad, R_edge


def CME_edge_ensemble(
    tt_array,
    r0,
    v0_array,
    gamma_array,
    w_array,
    omega_array,
    phi_cme_array,
    cone_geometry,
    kinematic,
):
    num_samples = 1000
    full_size = len(tt_array)
    assert all(
        len(arr) == full_size
        for arr in [v0_array, gamma_array, w_array, omega_array, phi_cme_array]
    ), "All input arrays must have same length"

    idx = np.random.choice(full_size, size=num_samples, replace=False)
    ensemble_edges = []
    ensemble_angle = []

    for i in idx:
        tt = tt_array[i]
        v0 = v0_array[i]
        gamma = gamma_array[i]
        w = w_array[i]
        omega = omega_array[i]
        phi_cme = phi_cme_array[i]

        angles_rad, R_edge = CME_edge(
            tt, r0, v0, gamma, w, omega, phi_cme, cone_geometry, kinematic
        )
        ensemble_edges.append(R_edge)
        ensemble_angle.append(angles_rad)
        # ic(angles_rad)

    ensemble_edges = np.array(ensemble_edges)  # shape: (num_samples, num_angles)
    ic(len(ensemble_angle))
    ic(np.shape(ensemble_edges))
    return ensemble_angle, ensemble_edges


def setup_heliosphere(
    arrival_UTC,
    r_target,
    tt,
    r0,
    v0,
    gamma,
    w,
    omega,
    phi_cme,
    phi_target,
    cone_geometry,
    kinematic,
    pdbm,
):
    AJD = Time(arrival_UTC).jd
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_axes([0.1, 0.1, 0.75, 0.75], projection="polar")

    # Planets
    mercury = Horizons(id="199", location="399", epochs=AJD)
    mercury_eph = mercury.ephemerides()
    mercury_hlon = mercury_eph["EclLon"][0]
    mercury_r = mercury_eph["r"][0]
    venus = Horizons(id="299", location="399", epochs=AJD)
    venus_eph = venus.ephemerides()
    venus_hlon = venus_eph["EclLon"][0]
    venus_r = venus_eph["r"][0]
    mars = Horizons(id="499", location="399", epochs=AJD)
    mars_eph = mars.ephemerides()
    mars_hlon = mars_eph["EclLon"][0]
    mars_r = mars_eph["r"][0]
    jupiter = Horizons(id="599", location="399", epochs=AJD)
    jupiter_eph = jupiter.ephemerides()
    jupiter_hlon = jupiter_eph["EclLon"][0]
    jupiter_r = jupiter_eph["r"][0]
    saturn = Horizons(id="699", location="399", epochs=AJD)
    saturn_eph = saturn.ephemerides()
    saturn_hlon = saturn_eph["EclLon"][0]
    saturn_r = saturn_eph["r"][0]
    uranus = Horizons(id="799", location="399", epochs=AJD)
    uranus_eph = uranus.ephemerides()
    uranus_hlon = uranus_eph["EclLon"][0]
    uranus_r = uranus_eph["r"][0]
    neptune = Horizons(id="899", location="399", epochs=AJD)
    neptune_eph = neptune.ephemerides()
    neptune_hlon = neptune_eph["EclLon"][0]
    neptune_r = neptune_eph["r"][0]
    # Earth
    earth = ephem.Sun()  # special rule: ask the hlon of the Sun you get the Earth
    earth.compute(arrival_UTC)
    earth_hlon = np.rad2deg(earth.hlon)
    earth_r = 1.0

    # Spacecrafts

    # Parker Solar Probe
    try:
        PSP = Horizons(id="-96", location="399", epochs=AJD)
        PSP_eph = PSP.ephemerides()
        PSP_hlon = PSP_eph["EclLon"][0]
        PSP_r = PSP_eph["r"][0]
        ax.plot(
            np.deg2rad(PSP_hlon),
            PSP_r,
            "d",
            markerfacecolor="black",
            markeredgecolor="black",
            label="Parker SP",
            markersize=10,
            linewidth=6,
        )
    except:
        print("NO PSP")

    # Solar Orbiter
    try:
        SolO = Horizons(id="-144", location="399", epochs=AJD)
        SolO_eph = SolO.ephemerides()
        SolO_hlon = SolO_eph["EclLon"][0]
        SolO_r = SolO_eph["r"][0]
        ax.plot(
            np.deg2rad(SolO_hlon),
            SolO_r,
            "d",
            markerfacecolor="blue",
            markeredgecolor="black",
            label="Solar Orbiter",
            markersize=10,
            linewidth=6,
        )

    except:
        print("NO SolO")

    # Bepi Colombo
    try:
        BepCo = Horizons(id="-121", location="399", epochs=AJD)
        BepCo_eph = BepCo.ephemerides()
        BepCo_hlon = BepCo_eph["EclLon"][0]
        BepCo_r = BepCo_eph["r"][0]
        ax.plot(
            np.deg2rad(BepCo_hlon),
            BepCo_r,
            "d",
            markerfacecolor="grey",
            markeredgecolor="black",
            label="Bepicolombo",
            markersize=10,
            linewidth=6,
        )

    except:
        print("NO Bepi Colombo")

    # STEREO-A
    try:
        STA = Horizons(id="-234", location="399", epochs=AJD)
        STA_eph = STA.ephemerides()
        STA_hlon = STA_eph["EclLon"][0]
        STA_r = STA_eph["r"][0]
        ax.plot(
            np.deg2rad(STA_hlon),
            STA_r,
            "d",
            markerfacecolor="orange",
            markeredgecolor="black",
            label="STEREO-A",
            markersize=10,
            linewidth=6,
        )

    except:
        print("NO ST-A")

    # STEREO-B
    try:
        STB = Horizons(id="-235", location="399", epochs=AJD)
        STB_eph = STB.ephemerides()
        STB_hlon = STB_eph["EclLon"][0]
        STB_r = STB_eph["r"][0]
        ax.plot(
            np.deg2rad(STB_hlon),
            STB_r,
            "d",
            markerfacecolor="Red",
            markeredgecolor="black",
            label="STEREO-B",
            markersize=10,
            linewidth=6,
        )

    except:
        print("NO ST-B")

    # Main program for the plot

    Max_AU = 2.0
    if r_target < 2.0:
        Max_AU = 2.0
        planet_list = ["Mercury", "Venus", "Earth", "Mars"]
        Coord_list = [mercury_hlon, venus_hlon, earth_hlon, mars_hlon]
        distance_list = [mercury_r, venus_r, earth_r, mars_r]
        for this_planet, this_coord, this_dist in zip(
            planet_list, Coord_list, distance_list
        ):
            ax.plot(
                np.deg2rad(this_coord), this_dist, "o", label=this_planet, markersize=13
            )
    elif r_target < 6.0:
        Max_AU = 6.0
        planet_list = ["Mercury", "Venus", "Earth", "Mars", "Jupiter"]
        Coord_list = [mercury_hlon, venus_hlon, earth_hlon, mars_hlon, jupiter_hlon]
        distance_list = [mercury_r, venus_r, earth_r, mars_r, jupiter_r]
        for this_planet, this_coord, this_dist in zip(
            planet_list, Coord_list, distance_list
        ):
            ax.plot(
                np.deg2rad(this_coord), this_dist, "o", label=this_planet, markersize=13
            )
    elif r_target < 11.0:
        Max_AU = 11.0
        planet_list = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn"]
        Coord_list = [
            mercury_hlon,
            venus_hlon,
            earth_hlon,
            mars_hlon,
            jupiter_hlon,
            saturn_hlon,
        ]
        distance_list = [mercury_r, venus_r, earth_r, mars_r, jupiter_r, saturn_r]
        for this_planet, this_coord, this_dist in zip(
            planet_list, Coord_list, distance_list
        ):
            ax.plot(
                np.deg2rad(this_coord), this_dist, "o", label=this_planet, markersize=13
            )
    else:
        Max_AU = 31.0
        planet_list = [
            "Mercury",
            "Venus",
            "Earth",
            "Mars",
            "Jupiter",
            "Saturn",
            "Uranus",
            "Neptune",
        ]
        Coord_list = [
            mercury_hlon,
            venus_hlon,
            earth_hlon,
            mars_hlon,
            jupiter_hlon,
            saturn_hlon,
            uranus_hlon,
            neptune_hlon,
        ]
        distance_list = [
            mercury_r,
            venus_r,
            earth_r,
            mars_r,
            jupiter_r,
            saturn_r,
            uranus_r,
            neptune_r,
        ]
        for this_planet, this_coord, this_dist in zip(
            planet_list, Coord_list, distance_list
        ):
            ax.plot(
                np.deg2rad(this_coord), this_dist, "o", label=this_planet, markersize=10
            )

    # Main part to plot CME in Heliosphere
    if pdbm == True:
        ax.axvline(np.deg2rad(np.median(phi_cme + omega)), color="r", ls="-")
        ax.axvline(np.deg2rad(np.median(phi_cme - omega)), color="r", ls="-")
        theta_ensemble, r_ensemble = CME_edge_ensemble(
            tt, r0, v0, gamma, w, omega, phi_cme, cone_geometry, kinematic
        )
        for i in range(0, len(theta_ensemble)):
            ax.plot(theta_ensemble[i], r_ensemble[i], "r-", alpha=0.01)

    else:
        ax.axvline(np.deg2rad(phi_cme + omega), color="r", ls="-")
        ax.axvline(np.deg2rad(phi_cme - omega), color="r", ls="-")
        theta, r = CME_edge(
            tt, r0, v0, gamma, w, omega, phi_cme, cone_geometry, kinematic
        )
        ax.plot(theta, r, "r-", label="CME")

    ax.set_rticks(np.linspace(0.0, Max_AU, 5))
    ax.plot(0, 0, "*", markersize=25, color="orange", label="Sun")
    ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize=14)
    ax.plot(np.deg2rad(phi_target), r_target, "x", markersize=10)
    ax.text(np.deg2rad(phi_target), r_target, "Target", fontsize=12, va="top")
    ax.axvline(np.deg2rad(phi_target))

    # Save the plot to an in-memory buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=600, bbox_inches="tight")
    buffer.seek(0)  # Move to the start of the buffer
    plt.close()  # Close the plot to free resources

    return buffer
