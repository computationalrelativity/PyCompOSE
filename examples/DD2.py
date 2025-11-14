# PyCompOSE: manages CompOSE tables
# Copyright (C) 2022, David Radice <david.radice@psu.edu>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Tested using: https://compose.obspm.fr/eos/18 and DD2_hydro_30-Mar-2015.h5
#
# This example shows how to import/export CompOSE tables and compute the sound speed
#
# Instructions
# - Create a folder "DD2" at the same location as this script
# - Download https://compose.obspm.fr/eos/18 and place it in a folder "DD2"
# - Download EOS.tar from https://zenodo.org/record/4159620 to get DD2_hydro_30-Mar-2015.h5
# - Run the script

# %%
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

import os
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))
from compose.eos import Metadata, Table

# %%
md = Metadata(
    pairs = {
        0: ("e", "electron"),
        10: ("n", "neutro"),
        11: ("p", "proton"),
        4002: ("He4", "alpha particle"),
        3002: ("He3", "helium 3"),
        3001: ("H3", "tritium"),
        2001: ("H2", "deuteron")
    },
    quads = {
        999: ("N", "average nucleous")
    }
)
eos = Table(md)
eos.read(os.path.join(SCRIPTDIR, "DD2"), enforce_equal_spacing=True)

# %%
eos.compute_cs2(floor=1e-6)
eos.compute_abar()
eos.validate()
# Remove the highest temperature point
eos.restrict_idx(it1=-1)
eos.shrink_to_valid_nb()

# %%
eos.write_hdf5(os.path.join(SCRIPTDIR, "DD2", "DD2.h5"))
eos.write_athtab(os.path.join(SCRIPTDIR, "DD2", "DD2.athtab"))

# %% Take the lowest T slice of the EOS
eos_cold = eos.slice_at_t_idx(0)
# %% Find beta equilibrium
eos_cold = eos_cold.make_beta_eq_table()

# %%
eos_cold.write_hdf5(os.path.join(SCRIPTDIR, "DD2", "DD2_T0.1_beta.h5"))
eos_cold.write_lorene(os.path.join(SCRIPTDIR, "DD2", "DD2_T0.1_beta.lorene"))
eos_cold.write_number_fractions(os.path.join(SCRIPTDIR, "DD2", "DD2_T0.1_beta_Y.out"))

# %%
print("{} <= nb <= {}".format(eos.nb.min(), eos.nb.max()))
print("{} <= cs2 <= {}".format(eos.thermo["cs2"].min(), eos.thermo["cs2"].max()))


# %% Load comparison table
c = 29979245800.0           # CGS
MeV = 1.60217733e-06        # CGS
fm = 1e-13                  # CGS

ref_table = h5py.File(os.path.join(SCRIPTDIR, "DD2", "DD2_hydro_30-Mar-2015.h5"), "r")
ref_t = np.array(ref_table["temperature"])
ref_ye = np.array(ref_table["ye"])
ref_mb = np.array(ref_table["mass_factor"])
ref_rho = np.array(ref_table["density"])
ref_cs2 = np.array(ref_table["cs2"])
ref_mb = ref_mb*MeV/c**2
ref_nb = ref_rho/(ref_mb)*fm**3


# %% Plot CompOSE EOS in pressure dominated regime
Y_e = 0.5
iyq = np.argmin(np.abs(eos.yq - Y_e))

plt.figure()
plt.plot(eos.t, eos.thermo["Q1"][0,iyq,:]/eos.t**4)
plt.xlabel(r"$T\ [{\rm MeV}]$")
plt.ylabel(r"$p/(n_b T^4)\ [{\rm MeV}^3]$")
plt.xscale("log")
plt.yscale("log")


# %% Compare CompOSE and stallarcollapse.org cs2 at zero density
Y_e = 0.5
iyc = np.argmin(np.abs(eos.yq - Y_e))
iys = np.argmin(np.abs(ref_ye - Y_e))

nb = 1e-8
inc = np.argmin(np.abs(eos.nb - nb))
ins = np.argmin(np.abs(ref_nb - nb))

plt.figure()
plt.plot(ref_t[:], ref_cs2[iys,:,ins], linewidth=3, label="stellarcollapse.org")
plt.plot(eos.t, eos.thermo["cs2"][inc,iyc,:], label="CompOSE")
plt.xlabel(r"$T\ [{\rm MeV}]$")
plt.ylabel(r"$c_s^2/c^2$")
plt.axhline(1/3, color='k', label=r"$1/3$")
plt.legend()
plt.xscale("log")
plt.yscale("log")


# %% Compare CompOSE and stallarcollapse.org cs2 at fixed temperature
Y_e = 0.1
iyc = np.argmin(np.abs(eos.yq - Y_e))
iys = np.argmin(np.abs(ref_ye - Y_e))

T = 10.0
itc = np.argmin(np.abs(eos.t - T))
its = np.argmin(np.abs(ref_t - T))

plt.figure()
plt.title("T = {:.2f}, Y_q = {:.2f}".format(eos.t[itc], eos.yq[iyc]))
plt.plot(ref_nb, ref_cs2[iys,its,:], linewidth=3, label="stellarcollapse.org")
plt.plot(eos.nb, eos.thermo["cs2"][:,iyc,itc], label="CompOSE")
plt.xlabel(r"$nb\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$c_s^2/c^2$")
plt.legend()
plt.xscale("log")
plt.yscale("log")


# %% Plot sound speed for fixed Y_e
Y_e = 0.1
iyc = np.argmin(np.abs(eos.yq - Y_e))
iys = np.argmin(np.abs(ref_ye - Y_e))

plt.figure()
plt.title("CompOSE - Y_q = {:.2f}".format(eos.yq[iyc]))
plt.pcolormesh(eos.nb, eos.t, eos.thermo["cs2"][:,iyc,:].T,cmap='jet', norm=LogNorm(1e-2,1))
plt.colorbar(label=r"$c_s^2$")
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$T\ [{\rm MeV}]$")
plt.xscale("log")
plt.yscale("log")

plt.figure()
plt.title("stellarcollapse.org - Y_e = {:.2f}".format(ref_ye[iys]))
plt.pcolormesh(ref_nb, ref_t, ref_cs2[iys,:,:],cmap='jet', norm=LogNorm(1e-2,1))
plt.colorbar(label=r"$c_s^2$")
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$T\ [{\rm MeV}]$")
plt.xscale("log")
plt.yscale("log")


# %% Check beta equilibrium
ref_eq_rho, ref_eq_Y_e = np.loadtxt(os.path.join(SCRIPTDIR, "DD2", "DD2_22-Jun-2018.pizza"),
        usecols=(0,6), skiprows=5, unpack=True)
ref_eq_rho /= 1e3

plt.figure()
plt.title("beta-equilibrium")
plt.plot(ref_eq_rho, ref_eq_Y_e, label="eos_tools", linewidth=3)
plt.plot(eos_cold.mn*eos_cold.nb*Table.unit_dens, eos_cold.Y["e"].flatten(), label="PyCompOSE")
plt.xscale("log")
plt.xlabel(r"$\rho [{\rm g}\ {\rm cm}^{-3}]$")
plt.ylabel(r"$Y_e^\beta$")
plt.legend()


# %% Interpolate to a range in the table
nb = 10.0**np.linspace(-5, 0.0, 5)
t  = 10.0**np.linspace(-1, 2, 10)
yq = np.linspace(eos.yq[0] ,eos.yq[-1], 3)
eos_interp = eos.interpolate(nb, yq, t, method="linear")
eos_interp.write_hdf5(os.path.join(SCRIPTDIR, "DD2", "DD2_tiny.h5"))

# %% Print the EOS at one point
log_P = np.log(eos_interp.thermo["Q1"]*nb[:,np.newaxis,np.newaxis])
nb_pt = np.array([0.5])
yq_pt = np.array([0.3])
T_pt = np.array([10.0])
P_interp = np.exp(eos_interp.eval_given_rtx(log_P, nb_pt, yq_pt, T_pt))
print("P(n = {:.2f}, yq = {:.2f}, T = {}) = {:.15e} [MeV fm^-3]".format(
    nb_pt[0], yq_pt[0], T_pt[0], P_interp[0]))


# %%
Y_e = 0.1
iy0 = np.argmin(np.abs(eos.yq - Y_e))
iy1 = np.argmin(np.abs(eos_interp.yq - Y_e))

T = 10.0
it0 = np.argmin(np.abs(eos.t - T))
it1 = np.argmin(np.abs(eos_interp.t - T))

plt.figure()
plt.plot(eos.nb, eos.thermo["Q1"][:,iy0,it0]*eos.nb, label="Original", linewidth=3)
plt.plot(eos_interp.nb, eos_interp.thermo["Q1"][:,iy1,it1]*eos_interp.nb, label="Interp")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$p\ [{\rm MeV}\ {\rm fm}^{-3}]$")

plt.figure()
plt.plot(eos.nb, eos.thermo["cs2"][:,iy0,it0], label="Original", linewidth=3)
plt.plot(eos_interp.nb, eos_interp.thermo["cs2"][:,iy1,it1], label="Interp")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel(r"$n_b\ [{\rm fm}^{-3}]$")
plt.ylabel(r"$c_s^2/c^2$")


# %%
plt.show()
