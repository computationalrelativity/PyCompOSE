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
from scipy.interpolate import interp1d
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
                    elif i1 == len(arr)-1:
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
    return interp1d(x, y, kind='cubic', bounds_error=True, **kwargs)

def find_temp_given_ent(t, yq, S, S0, options={'xatol': 1e-2, 'maxiter': 100}):
    """
    Find the temperature such that S(T, Yq) = S0 for each ye

    * t  : 1d grid of temperatures
    * yq : 1d grid of Ye
    * S  : 2d array of entropy S[iye,itemp]
    * S0 : wanted entropy

    options are passed to `scipy.optimize.minimize_scalar`
    """
    tout = np.zeros_like(ye1d)
    for iyq in range(yq.shape[0]):
        f = interpolator(t, (S[iyq,:] - S0)**2)
        res = minimize_scalar(f, bounds=(t[0], t[-1]), method='bounded',
            options=options)
        tout[iyq] = res.x
    return tout

def find_beta_eq(yq, mu_l, options={'xatol': 1e-6, 'maxiter': 100}):
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
    res = minimize_scalar(f, bounds=(yq[0], yq[-1]), method='bounded',
    options=options)
    return res.x