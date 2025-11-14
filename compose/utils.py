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

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator
from scipy.optimize import minimize_scalar


def find_valid_region(arr):
    """
    Utility function.

    Finds the largest contiguous range (i0, i1) in which
        np.all(arr[i0:i1]) == True
    """

    def regions(arr):
        i0 = 0
        while i0 < len(arr):
            if arr[i0] == True:
                for i1 in range(i0, len(arr)):
                    if arr[i1] == False:
                        yield (i0, i1)
                        i0 = i1 + 1
                        break
                    elif i1 == len(arr) - 1:
                        yield (i0, len(arr))
                        i0 = len(arr)
                        break
            else:
                i0 += 1

    rlist = list(regions(arr))
    irmax, lrmax = 0, 0
    for ir, r in enumerate(rlist):
        myl = r[1] - r[0]
        if myl > lrmax:
            lrmax = myl
            irmax = ir

    return rlist[irmax]


def interpolator(x, y, **kwargs):
    """
    Custom 1d interpolator
    """
    return CubicSpline(x, y, **kwargs)


def find_temp_given_ent(t, yq, S, S0, options={"xatol": 1e-2, "maxiter": 100}):
    """
    Find the temperature such that S(T, Yq) = S0 for each ye

    * t  : 1d grid of temperatures
    * yq : 1d grid of Ye
    * S  : 2d array of entropy S[iye,itemp]
    * S0 : wanted entropy

    options are passed to `scipy.optimize.minimize_scalar`
    """
    tout = np.zeros_like(yq)
    for iyq in range(yq.shape[0]):
        f = interpolator(t, (S[iyq, :] - S0) ** 2)
        res = minimize_scalar(
            f, bounds=(t[0], t[-1]), method="bounded", options=options
        )
        tout[iyq] = res.x
    return tout


def find_beta_eq(yq, mu_l, options={"xatol": 1e-6, "maxiter": 100}):
    """
    Find the neutrino-less beta equilibrium ye for each point
    in a 1D table tab(ye).

    Beta equilibrium is found from the condition

    mu_l = 0

    * yq   : charge fraction
    * mu_l : lepton chemical potential as a function of yq

    options are passed to `scipy.optimize.minimize_scalar`
    """
    # These cases have beta-equilibrium out of the table
    if np.all(mu_l > 0):
        return yq[0]
    if np.all(mu_l < 0):
        return yq[-1]

    f = interpolator(yq, mu_l**2)
    res = minimize_scalar(f, bounds=(yq[0], yq[-1]), method="bounded", options=options)
    return res.x


def read_micro_composite_index(Ki):
    """
    Convert a composite index Ki into a tuple (name, desc) for eos.micro quantites.
    """

    # Abridged copy of table 3.3 in CompOSE manual v3
    dense_matter_fermions = {
        0: ("e", "electron"),
        1: ("mu", "muon"),
        10: ("n", "neutron"),
        11: ("p", "proton"),
    }

    # Abridged copy of table 7.5 in CompOSE manual v3
    microscopic_quantites = {
        40: (
            "mL_{0:s}",
            "Effective {1:s} Landau mass with respect to particle mass: mL_{0:s} / m_{0:s} []",
        ),
        41: (
            "mD_{0:s}",
            "Effective {1:s} Dirac mass with respect to particle mass: mD_{0:s} / m_{0:s} []",
        ),
        50: (
            "U_{0:s}",
            "Non-relativistic {1:s} single-particle potential: U_{0:s} [MeV]",
        ),
        51: ("V_{0:s}", "Relativistic {1:s} vector self-energy: V_{0:s} [MeV]"),
        52: ("S_{0:s}", "Relativistic {1:s} scalar self-energy: S_{0:s} [MeV]"),
    }

    Ii = Ki // 1000
    Ji = Ki - 1000 * Ii

    particle_names = dense_matter_fermions[Ii]
    variable_symbol, variable_description = microscopic_quantites[Ji]
    variable_names = (
        variable_symbol.format(*particle_names),
        variable_description.format(*particle_names),
    )

    return variable_names


def convert_to_NQTs(fname_in, fname_out, NQT_order=2, use_bithacks=True):
    import h5py

    # Switching for different NQT forms
    if NQT_order == 1 and use_bithacks:
        from .NQTs.NQTLib import NQT_exp2_O1 as NQT_exp
        from .NQTs.NQTLib import NQT_log2_O1 as NQT_log
    elif NQT_order == 2 and use_bithacks:
        from .NQTs.NQTLib import NQT_exp2_O2 as NQT_exp
        from .NQTs.NQTLib import NQT_log2_O2 as NQT_log
    if NQT_order == 1 and not use_bithacks:
        from .NQTs.NQTLib import NQT_exp2_ldexp_O1 as NQT_exp
        from .NQTs.NQTLib import NQT_log2_frexp_O1 as NQT_log
    elif NQT_order == 2 and not use_bithacks:
        from .NQTs.NQTLib import NQT_exp2_ldexp_O2 as NQT_exp
        from .NQTs.NQTLib import NQT_log2_frexp_O2 as NQT_log

    table_h5_in = h5py.File(fname_in, "r")
    table_h5_out = h5py.File(fname_out, "w")

    # These are the necessary datasets for evolution.
    # These and only these will be converted and copied.
    # They must all be present in the input file.
    """
    [
    'Q1',
    'Q2',
    'Q3',
    'Q4',
    'Q5',
    'Q6',
    'Q7',
    'cs2',
    'mn',
    'mp',
    'nb',
    't',
    'yq'
    ]
    """

    # These datasets can be copied directly.
    dsets_to_copy = ["mn", "mp", "yq"]
    for key in dsets_to_copy:
        table_h5_in.copy(table_h5_in[key], table_h5_out, key)

    # Thses datasets need interpolation onto the new grid, but are otherwise unchanged.
    dsets_to_interp = ["Q2", "Q3", "Q4", "Q5", "Q6", "cs2"]

    # Set which datasets will use log-space interpolation.
    log_data = {}
    for key in dsets_to_interp:
        log_data[key] = False
    log_data["cs2"] = True

    # Get the shape of the data from Q1
    input_shape = np.array(table_h5_in["Q1"].shape)

    # Determine the dimensionality of the table
    dims = np.sum(input_shape != 1)

    # Only support for 1D and 3D tables is present
    assert (
        dims == 1 or dims == 3
    ), "convert_to_NQTs() only supports 1- and 3-dimensional tables."

    # Only support for 1D tables in rho is present
    if dims == 1:
        assert (
            input_shape[0] > 1
        ), "convert_to_NQTs() only supports 1-dimensional tables in rho."

    # Set up grid for interpolation
    nb_min = table_h5_in["nb"][0] * (1 + 1e-15)
    nb_max = table_h5_in["nb"][-1] * (1 - 1e-15)
    nb_new = NQT_exp(
        np.linspace(NQT_log(nb_min), NQT_log(nb_max), num=table_h5_in["nb"].shape[0])
    )

    if dims == 3:
        t_min = table_h5_in["t"][0] * (1 + 1e-15)
        t_max = table_h5_in["t"][-1] * (1 - 1e-15)
        t_new = NQT_exp(
            np.linspace(NQT_log(t_min), NQT_log(t_max), num=table_h5_in["t"].shape[0])
        )
    elif dims == 1:
        t_new = table_h5_in["t"]

    table_h5_out.create_dataset("nb", data=nb_new)
    table_h5_out.create_dataset("t", data=t_new)

    log_nb_old = np.log(table_h5_in["nb"])
    log_nb_new = np.log(nb_new)

    log_t_old = np.log(table_h5_in["t"])
    log_t_new = np.log(t_new)

    if dims == 3:
        interp_x_old = (log_nb_old, log_t_old)
        MG_log_nb_new, MG_log_t_new = np.meshgrid(log_nb_new, log_t_new, indexing="ij")
        interp_X_new = np.array([MG_log_nb_new.flatten(), MG_log_t_new.flatten()]).T
    elif dims == 1:
        interp_x_old = (log_nb_old,)
        interp_X_new = log_nb_new

    # Interpolate to new grid
    for key in dsets_to_interp:
        data_old = np.array(table_h5_in[key])
        data_new = np.zeros((nb_new.shape[0], data_old.shape[1], t_new.shape[0]))

        for yq_idx in range(data_old.shape[1]):
            data_current = data_old[:, yq_idx, :]
            if log_data[key]:
                data_current = np.log(data_current)
            interp_current = RegularGridInterpolator(
                interp_x_old, data_current, method="linear"
            )
            data_result = interp_current(interp_X_new).reshape(
                (data_new.shape[0], data_new.shape[2])
            )
            if log_data[key]:
                data_result = np.exp(data_result)
            data_new[:, yq_idx, :] = data_result

        table_h5_out.create_dataset(key, data=data_new)

    # For Q1 and Q7 we interpolate pressure and energy, then calculate Q1 and Q7 from those
    press_old = (np.array(table_h5_in["Q1"])) * (
        np.array(table_h5_in["nb"])[:, np.newaxis, np.newaxis]
    )
    energy_old = (
        ((np.array(table_h5_in["Q7"])) + 1)
        * ((np.array(table_h5_in["nb"]))[:, np.newaxis, np.newaxis])
        * (table_h5_in["mn"][()])
    )

    press_new = np.zeros((nb_new.shape[0], data_old.shape[1], t_new.shape[0]))
    energy_new = np.zeros((nb_new.shape[0], data_old.shape[1], t_new.shape[0]))

    # Do pressure and energy interpolation
    for yq_idx in range(data_old.shape[1]):
        press_current = press_old[:, yq_idx, :]
        energy_current = energy_old[:, yq_idx, :]

        press_interp_current = RegularGridInterpolator(
            interp_x_old, np.log(press_current), method="linear"
        )
        energy_interp_current = RegularGridInterpolator(
            interp_x_old, np.log(energy_current), method="linear"
        )

        press_result = press_interp_current(interp_X_new).reshape(
            (press_new.shape[0], press_new.shape[2])
        )
        energy_result = energy_interp_current(interp_X_new).reshape(
            (energy_new.shape[0], energy_new.shape[2])
        )

        press_new[:, yq_idx, :] = np.exp(press_result)
        energy_new[:, yq_idx, :] = np.exp(energy_result)

    # Calculate Q1 and Q7
    Q1_new = press_new / (nb_new[:, np.newaxis, np.newaxis])
    Q7_new = (
        energy_new / ((nb_new[:, np.newaxis, np.newaxis]) * (table_h5_out["mn"][()]))
    ) - 1

    table_h5_out.create_dataset("Q1", data=Q1_new)
    table_h5_out.create_dataset("Q7", data=Q7_new)

    # Report to user
    print("Datasets created:")
    print(table_h5_out.keys())

    # Finish up
    table_h5_in.close()
    table_h5_out.close()

    return None


def F2_Takahashi(eta):
    """
    Numpy-compatible Fermi-Integral of order 2, approximation by
    Takahashi, El Eid, Hillebrandt, A&A 67, 185 (1978)
    """

    eta = np.array(eta, dtype=float)  # convert eta in numpy array
    result = np.zeros_like(eta)  # creates array of same shape
    threshold = 1e-3

    # Boolean masks based on the threshold to filter the two regimes
    mask_neg = eta <= threshold
    mask_pos = ~mask_neg

    # eta <= threshold
    if np.any(mask_neg):
        e = np.exp(eta[mask_neg])
        num = 2.0 * e
        den = 1.0 + 0.1092 * np.exp(0.8908 * eta[mask_neg])
        result[mask_neg] = num / den

    # eta > threshold
    if np.any(mask_pos):
        et = eta[mask_pos]
        num = (et**3) / 3.0 + 3.2899 * et
        den = 1.0 - np.exp(-1.8246 * et)
        result[mask_pos] = num / den

    return result


def F3_Takahashi(eta):
    """
    Numpy-compatible Fermi-Integral of order 3, approximation by
    Takahashi, El Eid, Hillebrandt, A&A 67, 185 (1978)
    """

    eta = np.array(eta, dtype=float)  # convert eta in numpy array
    result = np.zeros_like(eta)  # creates array of same shape
    threshold = 1e-3

    # Boolean masks based on the threshold to filter the two regimes
    mask_neg = eta <= threshold
    mask_pos = ~mask_neg

    # eta <= threshold
    if np.any(mask_neg):
        e = np.exp(eta[mask_neg])
        num = 6.0 * e
        den = 1.0 + 0.0559 * np.exp(0.9069 * eta[mask_neg])
        result[mask_neg] = num / den

    # eta > threshold
    if np.any(mask_pos):
        et = eta[mask_pos]
        num = 0.25 * (et**4) + 4.9348 * (et**2) + 11.3644
        den = 1.0 + np.exp(-1.9039 * et)
        result[mask_pos] = num / den

    return result


def F2_Fukushima(y):
    """
    Numpy-compatible Fermi-Integral of order 2, approximation by
    Fukushima, App Math Comput 259 (2015) 708–729
    """

    y = np.array(y, dtype=float)
    fd = np.zeros_like(y)

    x = -np.abs(y)

    xm2 = np.where(x < -2.0)
    xm0 = np.where(np.logical_and(x >= -2.0, x <= 0.0))
    yp0 = np.where(y > 0.0)

    ex = np.exp(x[xm2])
    t = ex * 7.38905609893065023

    fd[xm2] = ex * (
        2.0
        - ex
        * (
            1914.06748184935743
            + t
            * (
                273.085756700981399
                + t * (8.5861610217850095 + t * 0.0161890243763741414)
            )
        )
        / (
            7656.2699273974454
            + t * (1399.35442210906621 + t * (72.929152915475392 + t))
        )
    )

    s = -0.5 * x[xm0]
    t = 1.0 - s

    fd[xm0] = (
        2711.49678259128843
        + t
        * (
            1299.85460914884154
            + t
            * (
                222.606134197895041
                + t
                * (
                    172.881855215582924
                    + t
                    * (
                        112.951038040682055
                        + t
                        * (
                            24.0376482128898634
                            + t
                            * (
                                -2.68393549333878715
                                + t * (-2.14077421411719935 - t * 0.326188299771397236)
                            )
                        )
                    )
                )
            )
        )
    ) / (
        2517.1726659917047
        + s
        * (
            3038.7689794575778
            + s
            * (
                2541.7823512372631
                + s
                * (
                    1428.0589853413436
                    + s
                    * (
                        531.62378035996132
                        + s
                        * (
                            122.54595216479181
                            + s * (8.395768655115050 + s * (-3.9142702096919080 - s))
                        )
                    )
                )
            )
        )
    )

    fd[yp0] += y[yp0] * (3.28986813369645287 + 0.333333333333333333 * y[yp0] ** 2)

    return fd


def F3_Fukushima(y):
    """
    Numpy-compatible Fermi-Integral of order 3, approximation by
    Fukushima, App Math Comput 259 (2015) 708–729
    """

    y = np.array(y, dtype=float)
    fd = np.zeros_like(y)

    x = -np.abs(y)

    xm2 = np.where(x < -2.0)
    xm0 = np.where(np.logical_and(x >= -2.0, x <= 0.0))
    yp0 = np.where(y > 0.0)

    ex = np.exp(x[xm2])
    t = ex * 7.38905609893065023

    fd[xm2] = ex * (
        6.0
        - ex
        * (
            5121.6401850302408
            + t
            * (
                664.28706260743472
                + t * (19.0856927562699544 + t * 0.0410982603688952131)
            )
        )
        / (
            13657.7071600806539
            + t * (2136.54222460571183 + t * (92.376788603062645 + t))
        )
    )

    s = -0.5 * x[xm0]
    t = 1.0 - s

    fd[xm0] = (
        7881.24597452900838
        + t
        * (
            4323.07526636309661
            + t
            * (
                1260.13125873282465
                + t
                * (
                    653.359212389160499
                    + t
                    * (
                        354.630774329461644
                        + t
                        * (
                            113.373708671587772
                            + t * (19.9559488532742796 + t * 1.59407954898394322)
                        )
                    )
                )
            )
        )
    ) / (
        2570.7250703533430
        + s
        * (
            2972.7443644211129
            + s
            * (
                2393.9995533270879
                + s
                * (
                    1259.0724833462608
                    + s
                    * (
                        459.86413596901097
                        + s * (112.60906419590854 + s * (16.468882811659000 + s))
                    )
                )
            )
        )
    )

    y2 = y[yp0] * y[yp0]
    fd[yp0] = -fd[yp0] + 11.3643939539669510 + y2 * (4.93480220054467931 + y2 * 0.25)

    return fd


def smoothstep3(x, down, up):
    """
    Smoothstep function that smoothly
    transitions from 0 to 1 between two points
    on the real axis. See e.g.
    https://en.wikipedia.org/wiki/Smoothstep.
    This is the 7th-order polynomial version,
    which guarantees differentiability up to
    the third derivative.

    Parameters
    down : start point of the transition, below the function equals 0 (float)
    up : end point of the transition, above the function equals 1. down < up mnust hold. (float)

    Returns:
    s : same shape as x, result of evaluating the function
    """

    assert down < up

    # Rescale the independent variable to [0, 1]
    x = (x - down) / (up - down)

    # Compute the smoothstep polynomial using Horner's method
    s = 35 * x**4 + x**5 * (-84 + x * (70 + x * (-20)))

    # Enforce the extremal values to be 0 and 1
    s[x < 0] = 0
    s[x > 1] = 1

    return s
