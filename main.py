from datetime import datetime
from icecream import ic

# Custom class for (P-)DBM calculation.
import dbm_functions as dbm
from calculation_1D import DBM1D
from calculation_2D import DBM2D

"""
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
"""
# %%

############################################
#   1D (P-)DBM
############################################

"""
For auto_dbm = True, values of w and $\gamma$ is taken from Mugatwla et al.,2024.
             = False, provided values of w and $\gamma$ is taken.

For P_DBM = True, uncertainty in CME observables will be considerd.
            User can provide value manualy or make it None.
            In case of None value, 10% uncertainty will be conisederd in observables. 
"""
dbm1D = DBM1D(
    time_utc=datetime(2022, 12, 1, 12, 23),  # time when CME is at r0
    r0=20,  # initial position [R_sun unit]
    v0=1000,  # speed at r0
    target_name="Earth",  # Name of target
    P_DBM=False,  # Make True if you want to use probabilistic approach
    auto_dbm=True,  # Make False if you want to provide value for (w,gamma)
    wind_type="Slow",  # Select solar wind type. Only if auto_dbm is set to True. ["Fast","Slow"]
    w=400,  # ambient solar wind speed [km/s]. In case of auto_dbm = True, value will be ignored.
    gamma=0.2e-7,  # drag parameter [km-1].In case of auto_dbm = True, value will be ignored.
    dt=20,  # uncertainty in time_utc [minute]. Ignored if P_DBM = False
    dr0=1,  # uncertainty in r0. Ignored if P_DBM = False [R_sun]
    dv0=100,  # uncertainty in v0. Ignored if P_DBM = False [km/s]
    dw=None,  # uncertainty in ambient solar wind spped [km/s]
    dgamma=None,  # uncertainty in drag parameter. [km-1]
)
A = dbm1D.P_DBM_run()
ic(A)


# %%


############################################
#   2D (P-)DBM
############################################


"""
For auto_dbm = True, values of w and $\gamma$ is taken from Mugatwla et al.,2024.
             = False, provided values of w and $\gamma$ is taken.

For P_DBM = True, uncertainty in CME observables will be considerd.
            User can provide value manualy or make it None.
            In case of None value, 10% uncertainty will be conisederd in observables. 

valid_cones = ["IC", "Ice-Cream Cone", "TC", "Tangential Cone", "CC", "Concentric Cone"]
        
valid_kinematic = ["SSE", "Self-Simmilar Expansion", "FCE", "Flattening Cone Evolution"]
"""

downloader = DBM2D(
    time_utc=datetime(2022, 12, 1, 12, 23),  # time when CME is at r0
    r0=20,  # initial position [R_sun unit]
    v0=1000,  # speed of CME at r0 [km/s]
    Omega=80,  # half angular width of CME cone
    Phi_CME=0,  # Central meridian of CME / CME propagation direction in storny hurst system
    cone_type="TC",  # type of cone geometry
    Kinematic="SSE",  # type of kinematic approach
    target_name="Earth",  # target name
    P_DBM=True,  # Make True if you want to use probabilistic approach
    auto_dbm=True,  # Make False if you want to provide value for (w,gamma)
    wind_type="Slow",  # Select solar wind type. Only if auto_dbm is set to True.
    w=400,  # ambient solar wind speed [km/s]. In case of auto_dbm = True, value will be ignored.
    gamma=0.2e-7,  # drag parameter [km-1].In case of auto_dbm = True, value will be ignored.
    dt=20,  # uncertainty in time_utc [minute]. Ignored if P_DBM = False
    dr0=1,  # uncertainty in r0. Ignored if P_DBM = False
    dv0=100,  # uncertainty in v0. Ignored if P_DBM = False [km/s]
    dw=None,  # uncertainty in ambient solar wind spped [km/s]
    dgamma=None,  # uncertainty in drag parameter. [km-1]
    domega=None,  # Uncertainty in omega [deg]
    dphi_cme=None,  # Uncertainty in Phi_CME
)
A = downloader.P_DBM_run()
ic(A)

# To see kinemtic plot
dbm.show_plots(A["RVT_plot"])
