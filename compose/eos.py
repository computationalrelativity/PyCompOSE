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

"""
Utilities to read general purpose (3D) EOS tables
"""

from copy import deepcopy
import h5py
from math import pi, floor
import numpy as np
import scipy.integrate as sint
import os
import sys
import struct

try:
    from ._version import version
except ImportError:
    version = "unknown"

class Metadata:
    """
    Class encoding the metadata/indexing used to read the EOS table

    Members

        thermo : list of extra quantities in the thermo table
        pairs  : dictionary of particle fractions in the compo table
        quad   : dictionary of isotope fractions in the compo table
    """
    def __init__(self, thermo=[], pairs={}, quads={}, micro={}):
        """
        Initialize the metadata

        * thermo : list of additional (EOS specific) thermo quantities
        * pairs  : additional particles
        * quads  : additional isotopes
        * micro  : microphysics quantites

        thermo is a list of tuples [(name, desc)]
        Other inputs are dictionaries of tuples {index: (name, desc)}
        """
        self.thermo = {
            1: ("Q1", "pressure over number density: p/nb [MeV]"),
            2: ("Q2", "entropy per baryon [kb]"),
            3: ("Q3", "scaled and shifted baryon chemical potential: mu_b/m_n - 1"),
            4: ("Q4", "scaled charge chemical potential: mu_q/m_n"),
            5: ("Q5", "scaled effective lepton chemical potential: mu_l/m_n"),
            6: ("Q6", "scaled free energy per baryon: f/(nb*m_n) - 1"),
            7: ("Q7", "scaled internal energy per baryon: e/(nb*m_n) - 1")
        }
        for ix in range(len(thermo)):
            self.thermo[ix + 8] = thermo[ix]

        self.pairs = pairs.copy()
        self.quads = quads.copy()
        self.micro = micro.copy()

class Table:
    """
    This class stores a table in CompOSE format.

    1D, 2D, and 3D tables are treated in the same way, with the only
    difference that some of the index ranges might be trivial.

    Data

        nb     : baryon number density [fm^-3]
        t      : temperature [MeV]
        yq     : charge fraction
        thermo : dictionary of 3D arrays containing the therm quantities
        Y      : dictionary of 3D arrays containing the number fractions
        A      : dictionary of 3D arrays containing the average mass of each isotope
        Z      : dictionary of 3D arrays containing the average charge of each isotope
        qK     : dictionary of 3D arrays containing the microphysics quantites

    Metadata

        mn, mp : neutron and proton mass [MeV]
        lepton : if True, then leptons are included in the EOS

    The indexing for the 3D arrays is

        inb, iyq, it

    That is, the temperature is the fastest running index.
    """

    """ multiply to convert MeV --> g """
    unit_mass = 1.7826619216277864e-27
    """ multiply to convert MeV --> K """
    unit_temp  = 1.0/8.617333262e-11
    """ multiply to convert MeV/fm^3 --> g/cm^3 """
    unit_dens  = 1.782662696e12
    """ multiply to convert MeV/fm^3 --> erg/cm^3 """
    unit_energy = 1.6021773299709372e33
    """ multiply to convert unitless specific internal energy --> erg/g"""
    unit_eps = 8.987551787368177e20
    """ multiply to convert MeV/fm^3 --> dyn/cm^2 """
    unit_press = 1.602176634e33

    def __init__(self, metadata:Metadata = Metadata(), dtype=np.float64):
        """
        Initialize an EOS object

        * metadata : machine readable version of the EOS data sheet
        * dtype : data type
        """
        self.md = metadata
        self.dtype = dtype

        self.nb = np.empty(0)
        self.t = np.empty(0)
        self.yq = np.empty(0)
        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])
        self.valid = np.zeros(self.shape, dtype=bool)

        self.mn = np.nan
        self.mp = np.nan
        self.lepton = False

        self.thermo = {}
        self.Y, self.A, self.Z = {}, {}, {}
        self.qK = {}
        self.lorene_cut = 0

        self.version = version
        self.git_hash = version.split("+g")[-1]

    def copy(self, copy_data=True):
        """
        Returns a copy of the table

        * copy_data : if False, only the grid and metadata are copied
        """
        eos = Table(self.md, self.dtype)
        eos.nb = self.nb.copy()
        eos.t = self.t.copy()
        eos.yq = self.yq.copy()
        eos.shape = deepcopy(self.shape)
        eos.valid = self.valid.copy()
        eos.mn = self.mn
        eos.mp = self.mp
        eos.lepton = self.lepton

        if copy_data:
            for key, data in self.thermo.items():
                eos.thermo[key] = data.copy()
            for key, data in self.Y.items():
                eos.Y[key] = data.copy()
            for key, data in self.A.items():
                eos.A[key] = data.copy()
            for key, data in self.Z.items():
                eos.Z[key] = data.copy()
            for key, data in self.qK.items():
                eos.qK[key] = data.copy()

        return eos

    def copy_along_yq(self,yq_new):
        """
        Clone 1-D or 2-D along new yq axis.
        """

        assert self.shape[1]==1

        eos_new = Table(self.md, self.dtype)
        eos_new.nb = self.nb.copy()
        eos_new.yq = yq_new
        eos_new.t = self.t.copy()
        eos_new.shape = (self.nb.shape[0], yq_new.shape[0], self.t.shape[0])
        eos_new.valid = np.full(eos_new.shape,self.valid)
        eos_new.mn = self.mn
        eos_new.mp = self.mp
        eos_new.lepton = self.lepton

        for key, data in self.thermo.items():
            eos_new.thermo[key] = np.full(eos_new.shape,data)
        for key, data in self.Y.items():
            eos_new.Y[key] = np.full(eos_new.shape,data)
        for key, data in self.A.items():
            eos_new.A[key] = np.full(eos_new.shape,data)
        for key, data in self.Z.items():
            eos_new.Z[key] = np.full(eos_new.shape,data)
        for key, data in self.qK.items():
            eos_new.qK[key] = np.full(eos_new.shape,data)

        return eos_new
        



    def compute_cs2(self, floor=None):
        """
        Computes the square of the sound speed
        """
        P = self.thermo["Q1"]*self.nb[:,np.newaxis,np.newaxis]
        S = self.thermo["Q2"]
        u = self.mn*(self.thermo["Q7"] + 1)
        h = u + self.thermo["Q1"]

        if (S.min()<=0.0):
            S_ = S + 2*max(sys.float_info.min, abs(S.min()))
        else:
            S_ = S

        dPdn = P*self.diff_wrt_nb(np.log(P))

        if self.t.shape[0] > 1:
            dPdt = P*self.diff_wrt_t(np.log(P))
            dSdn = S_*self.diff_wrt_nb(np.log(S_))
            dSdt = S_*self.diff_wrt_t(np.log(S_))

            self.thermo["cs2"] = (dPdn - dSdn/dSdt*dPdt)/h
        else:
            self.thermo["cs2"] = dPdn/h

        if floor is not None:
            self.thermo["cs2"] = np.maximum(self.thermo["cs2"], floor)
        self.md.thermo[12] = ("cs2", "sound speed squared [c^2]")

    def compute_abar(self):
        """
        Computes the average mass number
        """
        self.md.micro[10] = ("Abar", "average mass number")
        self.qK["Abar"] = sum(
            self.Y[nuc] for nuc in self.Y if nuc not in ["e", "mu", "tau"]
        )
        mask = self.qK["Abar"] <= 0
        if np.any(mask):
            print(f"sum(Y) <= 0 for {mask.sum()} points")

        self.qK["Abar"][mask] = 1

        self.qK["Abar"] = 1.0 / self.qK["Abar"]
        if not np.all(phys := (self.qK["Abar"] >= 0.9999)):
            print(f"Unphysical Abar in {np.sum(~phys)} points")
        self.qK["Abar"] = np.clip(self.qK["Abar"], 1.0, None)

    def diff_wrt_nb(self, Q):
        """
        Differentiate a 3D variable w.r.t nb

        This function is optimized for log spacing for nb, but will work with any spacing
        """
        log_nb = np.log(self.nb[:,np.newaxis,np.newaxis])
        dQdn = np.empty_like(Q)
        dQdn[1:-1,...] = (Q[2:,...] - Q[:-2,...])/(log_nb[2:] - log_nb[:-2])
        dQdn[0,...] = (Q[1,...] - Q[0,...])/(log_nb[1] - log_nb[0])
        dQdn[-1,...] = (Q[-1,...] - Q[-2,...])/(log_nb[-1] - log_nb[-2])
        return dQdn/self.nb[:,np.newaxis,np.newaxis]

    def diff_wrt_t(self, Q):
        """
        Differentiate a 3D variable w.r.t T

        This function is optimized for log spacing for T, but will work with any spacing

        NOTE: You will get an error if you try to differentiate w.r.t to T a 1D table
        """
        log_t = np.log(self.t[np.newaxis,np.newaxis,:])
        dQdt = np.empty_like(Q)
        dQdt[...,1:-1] = (Q[...,2:] - Q[...,:-2])/(log_t[...,2:] - log_t[...,:-2])
        dQdt[...,0] = (Q[...,1] - Q[...,0])/(log_t[0,0,1] - log_t[0,0,0])
        dQdt[...,-1] = (Q[...,-1] - Q[...,-2])/(log_t[0,0,-1] - log_t[0,0,-2])
        return dQdt/self.t[np.newaxis,np.newaxis,:]

    def eval_given_rtx(self, var, nb, yq, t):
        """
        Interpolates a given thermodynamic variable at the wanted locations

        * var : a 3D array with the data to interpolate
        * nb  : a 1D array of density points
        * t   : a 1D array of temperature points
        * yq  : a 1D array of charge fraction points

        NOTE: This is not meant to be particularly efficient
        """
        from scipy.interpolate import RegularGridInterpolator

        assert nb.shape == t.shape == yq.shape

        my_lnb = np.log(self.nb)
        my_lt = np.log(self.t)
        func = RegularGridInterpolator((my_lnb, self.yq, my_lt), var)

        xi = np.column_stack((np.log(nb).flatten(), yq.flatten(), np.log(t).flatten()))
        out = func(xi).reshape(nb.shape)

        return out

    def get_bilby_eos_table(self):
        """
        Create a bilby TabularEOS object with the EOS

        NOTE: This only works for 1D tables
        """
        assert self.shape[1] == self.shape[2] == 1

        from bilby.gw.eos.eos import TabularEOS, conversion_dict

        # Energy density and pressure in CGS
        e  = Table.unit_dens*self.nb[:]*self.mn*(self.thermo["Q7"][:,0,0] + 1)
        p  = Table.unit_press*self.thermo["Q1"][:,0,0]*self.nb[:]

        # Convert to Bilby units (G = c = 1, 1 meter = 1)
        e = e/conversion_dict["density"]["cgs"]
        p = p/conversion_dict["pressure"]["cgs"]
        table = np.column_stack((p, e))

        return TabularEOS(table, sampling_flag=True)

    def get_bilby_eos_family(self, npts=500):
        """
        Creates a bilby EOSFamily (a TOV sequence) for the EOS

        * npts : number of points on the TOV sequence

        NOTE: This only works for 1D tables
        """
        assert self.shape[1] == self.shape[2] == 1
        from bilby.gw.eos import EOSFamily
        return EOSFamily(self.get_bilby_eos_table(), npts=npts)

    def integrate_tov(self, rhoc):
        """
        Integrates the TOV equation for given central densities

        * rhoc : central energy density in MeV/fm^3

        Returns an object with the following attributes

        * nb     : central density in 1/fm^3
        * rho    : central energy density in MeV/fm^3
        * p      : central pressure in MeV/fm^3
        * K      : compressibility dp/dnb at the center
        * mass   : mass in solar masses
        * rad    : radius in km
        * c      : compactness
        * k2     : Love number
        * lmbda  : Tidal deformability coefficient

        NOTE: This requires bilby to be available and works for 1D tables only
        """
        class TOV:
            pass

        assert self.shape[1] == self.shape[2] == 1

        from bilby.gw.eos.eos import IntegrateTOV, conversion_dict
        from .utils import interpolator

        if not hasattr(rhoc, "__len__"):
            rhoc = [ rhoc ]

        eos = self.get_bilby_eos_table()

        mass, radius, compact, k2love_number, tidal_deformability = [], [], [], [], []
        for rc in rhoc:
            rc = (Table.unit_dens/conversion_dict["density"]["cgs"])*rc
            tov_solver = IntegrateTOV(eos, rc)

            m, r, k2 = tov_solver.integrate_TOV()

            lmbda = 2./3. * k2 * (r/m)**5

            mass.append(m * conversion_dict["mass"]["m_sol"])
            radius.append(r * conversion_dict["radius"]["km"])
            compact.append(m/r)
            k2love_number.append(k2)
            tidal_deformability.append(lmbda)

        tov = TOV()
        tov.rho = np.array(rhoc)
        tov.mass = np.array(mass)
        tov.rad = np.array(radius)
        tov.c = np.array(compact)
        tov.k2 = np.array(k2love_number)
        tov.lmbda = np.array(tidal_deformability)

        nb_from_e = interpolator(self.nb[:]*self.mn*(self.thermo["Q7"][:,0,0] + 1), self.nb)
        tov.nb = nb_from_e(tov.rho)

        p_from_nb = interpolator(self.nb[:], self.thermo["Q1"][:,0,0]*self.nb[:])
        tov.p = p_from_nb(tov.nb)
        # K = 9*dp/dn
        tov.K = 9*p_from_nb(tov.nb, 1)

        return tov

    def interpolate(self, nb_new=None, yq_new=None, t_new=None, method="cubic"):
        """
        Generate a new table by interpolating the EOS to the given grid

        * nb : 1D array with all the number densities
        * yq : 1D array with all the charge fractions
        * t  : 1D array with all the temperatures

        * method : interpolation method, is passed to scipy.RegularGridInterpolator

        Wrapper around separate functions for different table dimensions.
        """

        if(self.shape[0] > 1 and self.shape[1] > 1 and self.shape[2] > 1):
            if not(nb_new):
                nb_new = self.nb
            if not(yq_new):
                yq_new = self.yq
            if not(t_new):
                t_new = self.t

            return self.interpolate_3D(nb_new, yq_new, t_new, method=method)
        elif(self.shape[0] > 1 and self.shape[1] == 1 and self.shape[2] == 1):
            return self.interpolate_1D(nb_new, method="cubic")
        else:
            raise ValueError("interpolation not implemented for current table dimensions.")


    def interpolate_3D(self, nb_new, yq_new, t_new, method="cubic"):
        """
        Generate a new table by interpolating the EOS to the given grid

        * nb : 1D array with all the number densities
        * yq : 1D array with all the charge fractions
        * t  : 1D array with all the temperatures

        * method : interpolation method, is passed to scipy.RegularGridInterpolator

        NOTE: this only works for 3D tables
        """
        assert self.shape[0] > 1
        assert self.shape[1] > 1
        assert self.shape[2] > 1

        from scipy.interpolate import RegularGridInterpolator

        eos = Table(self.md, self.dtype)
        eos.nb = nb_new.copy()
        eos.t = t_new.copy()
        eos.yq = yq_new.copy()
        eos.shape = deepcopy((nb_new.shape[0], yq_new.shape[0], t_new.shape[0]))
        eos.valid = np.ones(eos.shape, dtype=bool)
        eos.mn = self.mn
        eos.mp = self.mp
        eos.lepton = self.lepton

        log_nb = np.log(self.nb)
        log_t = np.log(self.t)

        log_nb_new, yq_new, log_t_new = np.meshgrid(
                np.log(nb_new), yq_new, np.log(t_new), indexing='ij')
        xi = np.column_stack((log_nb_new.flatten(), yq_new.flatten(),
                log_t_new.flatten()))

        def interp_var_to_grid(var3d, log=False):
            if log:
                myvar = np.log(var3d)
            else:
                myvar = var3d
            func = RegularGridInterpolator((log_nb, self.yq, log_t),
                    myvar, method=method)
            res = func(xi).reshape(eos.shape)
            if log:
                return np.exp(res)
            return res

        for key in self.thermo.keys():
            if key == "Q1":
                eos.thermo[key] = interp_var_to_grid(self.thermo[key], True)
            else:
                eos.thermo[key] = interp_var_to_grid(self.thermo[key])
        for key in self.Y.keys():
            eos.Y[key] = interp_var_to_grid(self.Y[key])
        for key in self.A.keys():
            eos.A[key] = interp_var_to_grid(self.A[key])
        for key in self.Z.keys():
            eos.Z[key] = interp_var_to_grid(self.Z[key])
        for key in self.qK.keys():
            eos.qK[key] = interp_var_to_grid(self.qK[key])

        return eos
    
    def interpolate_1D(self, nb_new, method="cubic"):
        """
        Generate a new table by interpolating the EOS to the given grid

        * nb : 1D array with all the number densities
        * yq : 1D array with all the charge fractions
        * t  : 1D array with all the temperatures

        * method : interpolation method, is passed to scipy.RegularGridInterpolator

        NOTE: this only works for 1D tables
        """
        assert self.shape[0] > 1
        assert self.shape[1] == 1
        assert self.shape[2] == 1

        from scipy.interpolate import RegularGridInterpolator

        eos = Table(self.md, self.dtype)
        eos.nb = nb_new.copy()
        eos.t = self.t.copy()
        eos.yq = self.yq.copy()
        eos.shape = deepcopy((eos.nb.shape[0], eos.yq.shape[0], eos.t.shape[0]))
        eos.valid = np.ones(eos.shape, dtype=bool)
        eos.mn = self.mn
        eos.mp = self.mp
        eos.lepton = self.lepton

        log_nb = np.log(self.nb)

        log_nb_new = np.log(nb_new)
        xi = log_nb_new

        def interp_var_to_grid(var3d, log=False):
            if log:
                myvar = np.log(var3d[:,0,0])
            else:
                myvar = var3d[:,0,0]
            func = RegularGridInterpolator((log_nb,),
                    myvar, method=method)
            res = func(xi).reshape(eos.shape)
            if log:
                return np.exp(res)
            return res

        for key in self.thermo.keys():
            if key == "Q1":
                eos.thermo[key] = interp_var_to_grid(self.thermo[key], True)
            else:
                eos.thermo[key] = interp_var_to_grid(self.thermo[key])
        for key in self.Y.keys():
            eos.Y[key] = interp_var_to_grid(self.Y[key])
        for key in self.A.keys():
            eos.A[key] = interp_var_to_grid(self.A[key])
        for key in self.Z.keys():
            eos.Z[key] = interp_var_to_grid(self.Z[key])
        for key in self.qK.keys():
            eos.qK[key] = interp_var_to_grid(self.qK[key])

        return eos

    def make_entropy_slice(self, f_ent, nb_min=None, nb_max=None):
        '''
        Create a new table in which entropy is specified as a function of density

        Remark: the new table will be invalidated
        '''
        from .utils import interpolator
        from scipy.optimize import minimize_scalar

        def interp_to_given_t(var3d, t_s):
            out = np.empty_like(t_s)
            for inb in range(var3d.shape[0]):
                for iy in range(var3d.shape[1]):
                    f = interpolator(self.t, var3d[inb,iy,:])
                    out[inb,iy,0] = f(t_s[inb,iy,0])
            return out
        
        # Restrict the range if necessary
        mask = self.nb >= 0
        if not nb_min == None:
            mask = self.nb >= nb_min
        if not nb_max == None:
            mask = (self.nb <= nb_max) & mask

        # Calculate yq
        # Calculate the 2d entropy table
        s_3d = self.thermo['Q2'][mask,:,:]
        # Estimate temperature from entropy
        t_eq = np.zeros((self.nb[mask].shape[0], self.yq.shape[0], 1),
                        dtype=self.dtype)
        for inb in range(len(self.nb[mask])):
            S0 = f_ent(self.nb[mask][inb])
            for iyq in range(len(self.yq)):
                f = interpolator(self.t, (s_3d[inb,iyq,:] - S0)**2)
                res = minimize_scalar(f, bounds=(self.t[0], self.t[-1]), method='bounded',
                                      options={'xatol':1e-2,'maxiter':100})
                t_eq[inb,iyq,0] = res.x

        eos = self.copy(copy_data=False)
        eos.t = np.zeros(1, dtype=self.dtype)
        eos.nb = self.nb[mask]
        eos.shape = (eos.nb.shape[0], 1, 1)

        for key in self.thermo.keys():
            temp = self.thermo[key][mask,:,:]
            eos.thermo[key] = interp_to_given_t(temp, t_eq)
        eos.md.thermo[13] = ("temp", "temperature in MeV")
        eos.thermo['temp'] = t_eq
        for key in self.Y.keys():
            temp = self.Y[key][mask,:,:]
            eos.Y[key] = interp_to_given_t(temp, t_eq)
        for key in self.A.keys():
            temp = self.A[key][mask,:,:]
            eos.A[key] = interp_to_given_t(temp, t_eq)
        for key in self.Z.keys():
            temp = self.Z[key][mask,:,:]
            eos.Z[key] = interp_to_given_t(temp, t_eq)
        for key in self.qK.keys():
            temp = self.qK[key][mask,:,:]
            eos.qK[key] = interp_to_given_t(temp, t_eq)

        return eos

    def make_hot_slice(self, f_t, f_ye, nb_min=None, nb_max=None):
        """
        Create a new table in which temperature and ye are specified as functions of rho

        Remark: the new table will be invalidated
        """
        from .utils import interpolator
        from scipy.optimize import minimize_scalar

        def interp_to_given_yp(var3d, yq_s):
            out = np.empty_like(yq_s)
            for inb in range(var3d.shape[0]):
                for it in range(var3d.shape[2]):
                    f = interpolator(self.yq, var3d[inb,:,it])
                    out[inb,0,it] = f(yq_s[inb,0,it])
            return out

        def interp_to_given_t(var3d, t_s):
            out = np.empty_like(t_s)
            for inb in range(var3d.shape[0]):
                for iy in range(var3d.shape[1]):
                    f = interpolator(self.t, var3d[inb,iy,:])
                    out[inb,iy,0] = f(t_s[inb,iy,0])
            return out

        # Restrict the range if necessary
        mask = self.nb >= 0
        if not nb_min == None:
            mask = self.nb >= nb_min
        if not nb_max == None:
            mask = (self.nb <=nb_max) & mask

        # Calculate yq
        yq_eq = np.zeros((self.nb[mask].shape[0], 1, self.t.shape[0]), dtype=self.dtype)
        for inb in range(len(self.nb[mask])):
            yq_eq[inb,0,:] = f_ye(self.nb[mask][inb])
        # Calculate the 2d entropy table
        s_2d = interp_to_given_yp(self.thermo['Q2'][mask,:,:], yq_eq)
        # Estimate temperature from entropy
        t_eq = np.zeros((self.nb[mask].shape[0], 1, 1), dtype=self.dtype)
        for inb in range(len(self.nb[mask])):
            t_eq[inb,0,0] = f_t(self.nb[mask][inb])
        #for inb in range(len(self.nb[mask])):
        #    S0 = f_ent(self.nb[mask][inb])
        #    f = interpolator(self.t, (s_2d[inb,0,:] - S0)**2)
        #    res = minimize_scalar(f, bounds=(self.t[0], self.t[-1]), method='bounded',
        #                          options={'xatol':1e-2,'maxiter':100})
        #    t_eq[inb,0,0] = res.x

        eos = self.copy(copy_data=False)
        eos.yq = np.zeros(1, dtype=self.dtype)
        eos.t = np.zeros(1, dtype=self.dtype)
        eos.nb = self.nb[mask]
        eos.shape = (eos.nb.shape[0], 1, 1)

        for key in self.thermo.keys():
            temp = interp_to_given_yp(self.thermo[key][mask,:,:], yq_eq)
            eos.thermo[key] = interp_to_given_t(temp, t_eq)
        eos.md.thermo[13] = ("temp", "temperature in MeV")
        eos.thermo['temp'] = t_eq
        for key in self.Y.keys():
            temp = interp_to_given_yp(self.Y[key][mask,:,:], yq_eq)
            eos.Y[key] = interp_to_given_t(temp, t_eq)
        # Add the lepton fraction to the table.
        if not 'e' in eos.Y.keys():
            eos.md.pairs[1] = ("e", "electron/charge/lepton fraction")
            eos.Y['e'] = interp_to_given_t(yq_eq, t_eq)
        for key in self.A.keys():
            temp = interp_to_given_yp(self.A[key][mask,:,:], yq_eq)
            eos.A[key] = interp_to_given_t(temp, t_eq)
        for key in self.Z.keys():
            temp = interp_to_given_yp(self.Z[key][mask,:,:], yq_eq)
            eos.Z[key] = interp_to_given_t(temp, t_eq)
        for key in self.qK.keys():
            temp = interp_to_given_yp(self.qK[key][mask,:,:], yq_eq)
            eos.qK[key] = interp_to_given_t(temp, t_eq)

        return eos

    def make_beta_eq_table(self):
        """
        Create a new table in which yq is set by beta equilibrium

        Remark the new table will be invalidated
        """
        from .utils import find_beta_eq
        from .utils import interpolator

        def interp_to_given_yp(var3d, yq_eq):
            out = np.empty_like(yq_eq)
            for inb in range(var3d.shape[0]):
                for it in range(var3d.shape[2]):
                    f = interpolator(self.yq, var3d[inb,:,it])
                    out[inb,0,it] = f(yq_eq[inb,0,it])
            return out

        yq_eq = np.zeros((self.nb.shape[0], 1, self.t.shape[0]), dtype=self.dtype)
        for inb in range(len(self.nb)):
            for it in range(len(self.t)):
                # This is divided by the neutron mass, but it does not matter
                mu_l = self.thermo["Q5"]
                yq_eq[inb, 0, it] = find_beta_eq(self.yq, mu_l[inb, :, it])

        eos = self.copy(copy_data=False)
        eos.yq = np.zeros(1, dtype=self.dtype)
        eos.shape = (eos.nb.shape[0], 1, eos.t.shape[0])

        for key in self.thermo.keys():
            eos.thermo[key] = interp_to_given_yp(self.thermo[key], yq_eq)
        for key in self.Y.keys():
            eos.Y[key] = interp_to_given_yp(self.Y[key], yq_eq)
        # Add the lepton fraction to the table.
        if not 'e' in eos.Y.keys():
            eos.md.pairs[1] = ("e", "electron/charge/lepton fraction")
            eos.Y['e'] = yq_eq
        for key in self.A.keys():
            try:
                eos.A[key] = interp_to_given_yp(self.A[key], yq_eq)
            except ValueError:
                print("Could not interpolate A[{}]".format(key))
                eos.A[key] = np.ones_like(yq_eq)
        for key in self.Z.keys():
            try:
                eos.Z[key] = interp_to_given_yp(self.Z[key], yq_eq)
            except ValueError:
                print("Could not interpolate Z[{}]".format(key))
                eos.Z[key] = np.ones_like(yq_eq)
        for key in self.qK.keys():
            eos.qK[key] = interp_to_given_yp(self.qK[key], yq_eq)

        return eos

    def find_lorene_rho_cut(self, threshold=0.5) -> int:
        """
        Find the density cut for the Lorene txt file
        Returns first index where rho/P * dPdrho > threshold
        """
        assert self.shape[1] == self.shape[2] == 1
        n = self.nb
        P = self.thermo["Q1"][:, 0, 0] * n
        dPdn = np.gradient(P, n)
        self.lorene_cut = (n/P*dPdn).searchsorted(.5)
        return self.lorene_cut

    def remove_photons(self):
        """
        Generate a new table without photons

        This takes care of removing photons from Q1, Q2, Q6, and Q7,
        but not from other quantities
        """
        nb = self.nb[:,np.newaxis,np.newaxis]
        t = self.t[np.newaxis,np.newaxis,:]

        # photon energy density [MeV fm^-3]
        e_ph = pi**2/15*t**4
        # photon pressure [MeV fm^-3]
        p_ph = 1/3*e_ph
        # photon free energy density [MeV fm^-3]
        f_ph = -p_ph
        # photon entropy density [fm^-3]
        s_ph = 4*pi**2/45*t**3

        eos = self.copy()

        p = self.thermo["Q1"]*nb
        eos.thermo["Q1"] = (p - p_ph)/nb

        s = self.thermo["Q2"]*nb
        eos.thermo["Q2"] = (s - s_ph)/nb

        f = self.mn*nb*(self.thermo["Q6"] + 1)
        eos.thermo["Q6"] = (f - f_ph)/(self.mn*nb) - 1

        e = self.mn*nb*(self.thermo["Q7"] + 1)
        eos.thermo["Q7"] = (e - e_ph)/(self.mn*nb) - 1

        return eos

    def enforce_pressure_temperature_monotonicity(self,logp=True,verb=0):
        nb = self.nb[:,np.newaxis,np.newaxis]
        p = self.thermo["Q1"]*nb
        if logp:
            p = np.log(p)

        eos_new = self.copy()

        if verb>0: print(np.sum((p[:,:,1:] - p[:,:,:-1])<0.0))
        if verb>0: print(np.sum((p[1:,:,:] - p[:-1,:,:])<0.0))

        for nb_idx in range(self.shape[0]):
            if verb>1: print(nb_idx,end="\r")
            for yq_idx in range(self.shape[1]):
                t_idx = 0
                while t_idx < self.shape[2]-1:
                    start_idx = t_idx
                    while p[nb_idx,yq_idx,t_idx+1]<=p[nb_idx,yq_idx,start_idx]:
                        t_idx += 1
                    end_idx = t_idx + 1

                    if end_idx>start_idx+1:
                        if verb>2: print()
                        while np.any((p[nb_idx,yq_idx,start_idx+1:end_idx+1]-p[nb_idx,yq_idx,start_idx:end_idx])<0):
                            if verb>2: print(nb_idx,yq_idx,start_idx,end_idx,p[nb_idx,yq_idx,start_idx],p[nb_idx,yq_idx,end_idx],np.min(p[nb_idx,yq_idx,start_idx+1:end_idx+1]-p[nb_idx,yq_idx,start_idx:end_idx]),end="\r")
                            p[nb_idx,yq_idx,start_idx+1:end_idx] = (p[nb_idx,yq_idx,start_idx:end_idx-1] + p[nb_idx,yq_idx,start_idx+1:end_idx] + p[nb_idx,yq_idx,start_idx+2:end_idx+1])/3
                        if verb>2: print()
                    t_idx += 1

        if verb>0: print(np.sum((p[:,:,1:] - p[:,:,:-1])<0.0))
        if verb>0: print(np.sum((p[1:,:,:] - p[:-1,:,:])<0.0))

        if logp:
            p = np.exp(p)

        eos_new.thermo["Q1"] = p/nb

        return eos_new

    def enforce_pressure_density_monotonicity(self,logp=True,verb=0):
        nb = self.nb[:,np.newaxis,np.newaxis]
        p = self.thermo["Q1"]*nb
        if logp:
            p = np.log(p)

        eos_new = self.copy()

        if verb>0: print(self.shape)
        if verb>0: print(p.shape)
        if verb>0: print(np.sum((p[:,:,1:] - p[:,:,:-1])<0.0))
        if verb>0: print(np.sum((p[1:,:,:] - p[:-1,:,:])<0.0))

        for yq_idx in range(self.shape[1]):
            if verb>1: print(yq_idx,end="\r")
            for t_idx in range(self.shape[2]):
                nb_idx = 0
                while nb_idx < self.shape[0]-1:
                    start_idx = nb_idx
                    while p[nb_idx+1,yq_idx,t_idx]<=p[start_idx,yq_idx,t_idx]:
                        nb_idx += 1
                    end_idx = nb_idx + 1

                    if end_idx>start_idx+1:
                        if verb>2: print()
                        while np.any((p[start_idx+1:end_idx+1,yq_idx,t_idx]-p[start_idx:end_idx,yq_idx,t_idx])<0):
                            if verb>2: print(start_idx,end_idx,yq_idx,t_idx,p[start_idx,yq_idx,t_idx],p[end_idx,yq_idx,t_idx],np.min(p[start_idx+1:end_idx+1,yq_idx,t_idx]-p[start_idx:end_idx,yq_idx,t_idx]),end="\r")
                            p[start_idx+1:end_idx,yq_idx,t_idx] = (p[start_idx:end_idx-1,yq_idx,t_idx] + p[start_idx+1:end_idx,yq_idx,t_idx] + p[start_idx+2:end_idx+1,yq_idx,t_idx])/3
                        if verb>2: print()
                    nb_idx += 1

        if verb>0: print(np.sum((p[:,:,1:] - p[:,:,:-1])<0.0))
        if verb>0: print(np.sum((p[1:,:,:] - p[:-1,:,:])<0.0))

        if logp:
            p = np.exp(p)

        eos_new.thermo["Q1"] = p/nb

        if verb>0: print(eos_new.shape)
        if verb>0: print((p/nb).shape)

        return eos_new

    def restrict(self, nb_min=None, nb_max=None, yq_min=None, yq_max=None,
            t_min=None, t_max=None):
        """
        Restrict the table in the given range
        """
        if nb_min is not None:
            assert nb_min < self.nb[-1]
            in0 = self.nb.searchsorted(nb_min)
        else:
            in0 = None
        if nb_max is not None:
            in1 = self.nb.searchsorted(nb_max)
        else:
            in1 = None

        if yq_min is not None:
            assert yq_min < self.yq[-1]
            iy0 = self.yq.searchsorted(yq_min)
        else:
            iy0 = None
        if yq_max is not None:
            iy1 = self.yq.searchsorted(yq_max)
        else:
            iy1 = None

        if t_min is not None:
            assert t_min < self.t[-1]
            it0 = self.t.searchsorted(t_min)
        else:
            it0 = None
        if t_max is not None:
            it1 = self.t.searchsorted(t_max)
        else:
            it1 = None

        self.restrict_idx(in0, in1, iy0, iy1, it0, it1)

    def restrict_idx(self, in0=None, in1=None, iy0=None, iy1=None, it0=None, it1=None):
        """
        Restrict the table to a given indicial range
        """
        self.nb = self.nb[in0:in1]
        self.yq = self.yq[iy0:iy1]
        self.t = self.t[it0:it1]
        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])
        self.valid = self.valid[in0:in1,iy0:iy1,it0:it1]

        for key in self.thermo.keys():
            self.thermo[key] = self.thermo[key][in0:in1,iy0:iy1,it0:it1]
        for key in self.Y.keys():
            self.Y[key] = self.Y[key][in0:in1,iy0:iy1,it0:it1]
        for key in self.A.keys():
            self.A[key] = self.A[key][in0:in1,iy0:iy1,it0:it1]
        for key in self.Z.keys():
            self.Z[key] = self.Z[key][in0:in1,iy0:iy1,it0:it1]
        for key in self.qK.keys():
            self.qK[key] = self.qK[key][in0:in1,iy0:iy1,it0:it1]

    def get_polytrope(self,nb_idx):
        """
        Get the polytrope coefficients Gamma and Kappa (p=K*rho^G) at a given nb index.
        
        Only valid for 1D tables (constant T and Ye)
        """
        assert self.shape[0] > 1 and self.shape[1] == 1 and self.shape[2] == 1
        
        if nb_idx==0:
            nb = self.nb[0:3]
            press = self.thermo["Q1"][0:3,0,0]*nb
            
            log_nb = np.log(nb)
            log_press = np.log(press)
            
            dlpdlnb = (-3*log_press[0] + 4*log_press[1] - log_press[2])/(log_nb[2] - log_nb[0])
            lp = log_press[0]
            lnb = log_nb[0]
            
        elif nb_idx==-1 or nb_idx==self.shape[0]-1:
            nb = self.nb[-3:]
            press = self.thermo["Q1"][-3:,0,0]*nb
            
            log_nb = np.log(nb)
            log_press = np.log(press)
            
            dlpdlnb = (log_press[0] - 4*log_press[1] + 3*log_press[2])/(log_nb[2] - log_nb[0])
            lp = log_press[2]
            lnb = log_nb[2]
            
            
        else:
            nb = self.nb[nb_idx-1:nb_idx+2]
            press = self.thermo["Q1"][nb_idx-1:nb_idx+2,0,0]*nb
            
            log_nb = np.log(nb)
            log_press = np.log(press)
            
            dlpdlnb = (-1*log_press[0] + log_press[2])/(log_nb[2] - log_nb[0])
            lp = log_press[1]
            lnb = log_nb[1]
            
            
        Gamma = dlpdlnb
        Kappa = np.exp(lp - Gamma*(lnb + np.log(self.mn)))
        
        return Kappa, Gamma
    
    def extend_with_polytrope(self, nb_min, Kappa, Gamma):
        """
        Extend a 1D table down to nb_min using the polytrope given. 
        The original grid is assumed to be in equal log spacing of nb, and this grid is extended down to nb_min, so the final nb[0]>=nb_min.
        
        Q1, Q3, Q6, and Q7 are calculated, Q2 is set to zero, and Q4 and Q5 repeat their values at the lower edge of the existing table
        """
        log_nb = np.log(self.nb)
        log_nb_min = np.log(nb_min)
        dlog_nb = log_nb[1] - log_nb[0]
        new_nb_count = floor((log_nb[0]-log_nb_min)/dlog_nb)
        new_log_nb = np.arange(-new_nb_count,0)*dlog_nb + log_nb[0]
        new_nb = np.exp(new_log_nb)
        
        new_press = Kappa * ((self.mn*new_nb)**Gamma)
        
        new_eps_shifted = (Kappa/(Gamma-1))*((self.mn*new_nb)**(Gamma-1))
        new_eps_0 = (Kappa/(Gamma-1))*((self.mn*self.nb[0])**(Gamma-1))
        old_eps_0 = self.thermo["Q7"][0,0,0]
        new_eps_const = old_eps_0 - new_eps_0
        new_eps = new_eps_shifted + new_eps_const
        
        new_mub_scaled = self.mn*(1 + new_eps + (Gamma-1)*(new_eps - new_eps_const))
        new_mub_0 = self.mn*(1 + (new_eps_0+new_eps_const) + (Gamma-1)*new_eps_0)
        old_mub_0 = (self.thermo["Q3"][0,0,0]+1)*self.mn
        new_mub_scale = old_mub_0 / new_mub_0
        new_mub = new_mub_scaled * new_mub_scale
        
        new_thermo = {}
        new_thermo["Q1"] = new_press/new_nb
        new_thermo["Q2"] = np.zeros(new_nb_count)
        new_thermo["Q3"] = (new_mub/self.mn) - 1
        new_thermo["Q4"] = np.ones(new_nb_count)*self.thermo["Q4"][0,0,0]
        new_thermo["Q5"] = np.ones(new_nb_count)*self.thermo["Q5"][0,0,0]
        new_thermo["Q6"] = new_eps
        new_thermo["Q7"] = new_eps
        
        eos = Table(self.md, self.dtype)
        eos.nb = np.concatenate((new_nb,self.nb.copy()),axis=0)
        eos.t = self.t.copy()
        eos.yq = self.yq.copy()
        eos.shape = (new_nb_count + self.shape[0],self.shape[1],self.shape[2])
        eos.valid = np.zeros(eos.shape, dtype=bool)
        eos.mn = self.mn
        eos.mp = self.mp
        eos.lepton = self.lepton

        for key, data in self.thermo.items():
            if key=="cs2":
                continue
            eos.thermo[key] = np.concatenate((new_thermo[key][:,np.newaxis,np.newaxis],data),axis=0)
        # for key, data in self.Y.items():
        #     eos.Y[key] = data.copy()
        # for key, data in self.A.items():
        #     eos.A[key] = data.copy()
        # for key, data in self.Z.items():
        #     eos.Z[key] = data.copy()

        return eos

    def read(self, path, enforce_equal_spacing=False, log_idvars=(True,False,True)):
        """
        Read the table from CompOSE ASCII format

        * path : folder containing the EOS in CompOSE format
        """
        self.path = path

        self.nb = np.loadtxt(os.path.join(path, "eos.nb"), skiprows=2, dtype=self.dtype)
        self.t = np.loadtxt(os.path.join(path, "eos.t"), skiprows=2, dtype=self.dtype).reshape(-1)
        self.yq = np.loadtxt(os.path.join(path, "eos.yq"), skiprows=2, dtype=self.dtype).reshape(-1)

        ### some tables include 0-temperature slice at T index zero, this breaks log(T) interpolation, so we remove it here, but only if it is not the only T index
        if self.t[0] == 0 and self.t.shape[0]>1:
            self.t = self.t[1:None]

        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])
        self.valid = np.ones(self.shape, dtype=bool)

        if enforce_equal_spacing:
            nb_log, yq_log, t_log = log_idvars

            if nb_log:
                nb_min = np.log(self.nb[0])
                nb_max = np.log(self.nb[-1])
            else:
                nb_min = self.nb[0]
                nb_max = self.nb[-1]
            d_nb = (nb_max - nb_min)/(self.shape[0] - 1)

            for idx in range(1,self.shape[0]-1):
                if nb_log:
                    self.nb[idx] = np.exp(nb_min + idx*d_nb)
                else:
                    self.nb[idx] = nb_min + idx*d_nb

            ### skip this if there is only one yq
            if self.shape[1]!=1:
                if yq_log:
                    yq_min = np.log(self.yq[0])
                    yq_max = np.log(self.yq[-1])
                else:
                    yq_min = self.yq[0]
                    yq_max = self.yq[-1]
                d_yq = (yq_max - yq_min)/(self.shape[1] - 1)

                for idx in range(1,self.shape[1]-1):
                    if yq_log:
                        self.yq[idx] = np.exp(yq_min + idx*d_yq)
                    else:
                        self.yq[idx] = yq_min + idx*d_yq
            
            ### skip this if there is only one T
            if self.shape[2]!=1:
                if t_log:
                    t_min = np.log(self.t[0])
                    t_max = np.log(self.t[-1])
                else:
                    t_min = self.t[0]
                    t_max = self.t[-1]
                d_t = (t_max - t_min)/(self.shape[2] - 1)

                for idx in range(1,self.shape[2]-1):
                    if t_log:
                        self.t[idx] = np.exp(t_min + idx*d_t)
                    else:
                        self.t[idx] = t_min + idx*d_t

        L = open(os.path.join(path, "eos.thermo"), "r").readline().split()
        self.mn = float(L[0])
        self.mp = float(L[1])
        self.lepton = bool(L[2])

        self.__read_thermo_entries()

        if os.path.exists(os.path.join(self.path, "eos.compo")):
            self.__read_compo_entries()

        if os.path.exists(os.path.join(self.path, "eos.micro")):
            self.__read_micro_entries()

    def __indicies_from_line(self, L):
        skip = False
        it, inb, iyq = int(L[0])-1, int(L[1])-1, int(L[2])-1

        ### skip if T index is 0 (now -1), or adjust if 0 is the only T index.
        if it==-1:
            if self.shape[2]==1:
                it = 0
            else:
                skip=True
        
        ## sometimes the only yq index is 0 instead of 1 (the CompOSE documentation is agnostic towards this), adjust if necessary.
        if iyq==-1:
            assert self.shape[1]==0, "iyq=0 where indicies should start from 1"
            iyq=0
        
        return it, inb, iyq, skip

    def __read_thermo_entries(self):
        """
        Parse eos.thermo using the given metadata key
        """
        self.thermo = {}
        for name, desc in self.md.thermo.values():
            self.thermo[name] = np.empty(self.shape, dtype=self.dtype)
        with open(os.path.join(self.path, "eos.thermo"), "r") as tfile:
            _ = tfile.readline()
            for line in tfile:
                L = line.split()
                it, inb, iyq, skip = self.__indicies_from_line(L)
                if skip: continue
                
                for iv in range(1, 8):
                    self.thermo[self.md.thermo[iv][0]][inb, iyq, it] = \
                        float(L[2 + iv])
                Nadd = int(L[10])
                for iv in range(8, 8+Nadd):
                    if iv in self.md.thermo:
                        self.thermo[self.md.thermo[iv][0]][inb, iyq, it] = \
                            float(L[2 + 1 + iv])

    def __read_compo_entries(self):
        """
        Parse eos.compo using the given metadata key
        """
        self.Y, self.A, self.Z = {}, {}, {}
        for name, desc in self.md.pairs.values():
            self.Y[name] = np.zeros(self.shape, dtype=self.dtype)
        for name, desc in self.md.quads.values():
            self.Y[name] = np.zeros(self.shape, dtype=self.dtype)
            self.A[name] = np.zeros(self.shape, dtype=self.dtype)
            self.Z[name] = np.zeros(self.shape, dtype=self.dtype)
        with open(os.path.join(self.path, "eos.compo"), "r") as cfile:
            for line in cfile:
                L = line.split()
                it, inb, iyq, skip = self.__indicies_from_line(L)
                if skip: continue
                Nphase = int(L[3])
                Npairs = int(L[4])
                ix = 5
                for ip in range(Npairs):
                    I, Y = int(L[ix]), float(L[ix + 1])
                    ix += 2
                    if I in self.md.pairs:
                        self.Y[self.md.pairs[I][0]][inb, iyq, it] = Y
                Nquad = int(L[ix])
                ix += 1
                for iq in range(Nquad):
                    I, A, Z, Y = int(L[ix]), float(L[ix+1]), \
                                 float(L[ix+2]), float(L[ix+3])
                    ix += 4
                    if I in self.md.quads:
                        name = self.md.quads[I][0]
                        self.A[name][inb, iyq, it] = A
                        self.Z[name][inb, iyq, it] = Z
                        self.Y[name][inb, iyq, it] = Y

    def __read_micro_entries(self):
        """
        Parse eos.micro using the given metadata key
        """
        self.qK = {}
        for name, desc in self.md.micro.values():
            self.qK[name] = np.zeros(self.shape, dtype=self.dtype)
        with open(os.path.join(self.path, "eos.micro"), "r") as cfile:
            for line in cfile:
                L = line.split()
                it, inb, iyq, skip = self.__indicies_from_line(L)
                if skip: continue
                Nmicro = int(L[3])
                ix = 4
                for im in range(Nmicro):
                    K, q = int(L[ix]), float(L[ix + 1])
                    ix += 2
                    if K in self.md.micro:
                        self.qK[self.md.micro[K][0]][inb, iyq, it] = q

    def read_from_pizza(
        self,
        hydro_path,
        weak_path,
        m_for_mub=None,
    ):
        """
        This function reads the EOS from the pizza hydro and weak files and
        initializes a pycompose Table object. Pizza tables use a "custom" mass
        factor while CompOSE uses the neutron mass as default. The specific
        internal energy and the chemical potentials are rescaled such that they
        are compatible to the neutron mass.
        Empirically, it seems that the baryon mass was not rescaled in the pizza
        tables. To correct this, use m_for_mb=931.4941 (atomic mass unit).
        """
        with h5py.File(hydro_path, 'r') as hf:
            hydro = {key: np.array(hf[key][:]) for key in hf}
        with h5py.File(weak_path, 'r') as hf:
            weak = {key: np.array(hf[key][:]) for key in hf}

        for key, ar in {**hydro, **weak}.items():
            if not len(ar.shape) == 3:
                continue
            hydro[key] = np.transpose(ar, (2, 0, 1))
        del weak

        mb = hydro['mass_factor']

        if m_for_mub is None:
            m_for_mub = mb

        self.mn = 939.56535
        self.mp = 938.27209
        self.nb = hydro['density'] / Table.unit_dens/mb
        self.t = hydro['temperature']
        self.yq = hydro['ye']

        self.shape = (self.nb.shape[0], self.yq.shape[0], self.t.shape[0])
        self.valid = np.ones(self.shape, dtype=bool)
        self.lepton = True

        self.thermo["Q1"] = hydro['pressure'] / Table.unit_press / self.nb[:, None, None]
        self.thermo["Q2"] = hydro['entropy']
        mu_e = hydro["mu_e"]
        mu_p = hydro["mu_p"]
        mu_n = hydro["mu_n"]
        eps = hydro['internalEnergy'] / Table.unit_eps
        # transform to new mass factor
        eps = (1+eps)*mb/self.mn - 1
        self.thermo["Q7"] = eps
        temp_entr = self.t[None,  None, :]*self.thermo["Q2"]
        self.thermo["Q6"] = eps - temp_entr/self.mn

        self.thermo["Q3"] = (mu_n  + m_for_mub)/self.mn - 1
        self.thermo["Q4"] = (mu_p - mu_n)/self.mn
        self.thermo["Q5"] = (mu_e + mu_p - mu_n)/self.mn

        self.Y["e"] = np.meshgrid(self.nb, self.yq, self.t, indexing='ij')[1]
        self.Y["n"] = hydro['Xn']
        self.Y["p"] = hydro['Xp']
        self.Y["He4"] = hydro['Xa']/4
        # "Abar" in Pizza seems to refer to the A of the representative nucleus
        self.Y["N"] = hydro["Xh"]/hydro["Abar"]
        for name, _ in self.md.pairs.values():
            if name not in self.Y:
                self.Y[name] = np.zeros_like(self.Y['e'])

        self.A["N"] = hydro["Abar"]
        self.Z["N"] = hydro["Zbar"]

    def shrink_to_valid_nb(self):
        """
        Restrict the range of nb
        """
        from .utils import find_valid_region

        if np.all(self.valid):
            return

        valid_nb = np.all(self.valid, axis=(1,2))
        in0, in1 = find_valid_region(valid_nb)

        excl_str = []
        if in0 != 0:
            excl_str.append(f"0 - {in0}")
        if in1 != self.shape[0]:
            excl_str.append(f"{in1} - {self.shape[0]}")
        if len(excl_str) > 0:
            excl_str = " and i_n=".join(excl_str)
            print(f"removing i_n={excl_str} to nb-range {self.nb[in0]}-{self.nb[in1]}!")

        self.restrict_idx(in0=in0, in1=in1)

    def slice_at_y_idx(self, iy):
        """
        Constructs a new table at a fixed composition self.yq[iy]
        """
        eos = self.copy(copy_data=False)
        eos.yq = np.array(eos.yq[iy], dtype=self.dtype).reshape((-1))
        eos.shape = (eos.nb.shape[0], 1, eos.t.shape[0])

        for key in self.thermo.keys():
            eos.thermo[key] = self.thermo[key][:,iy,:].reshape(eos.shape)
        for key in self.Y.keys():
            eos.Y[key] = self.Y[key][:,iy,:].reshape(eos.shape)
        for key in self.A.keys():
            eos.A[key] = self.A[key][:,iy,:].reshape(eos.shape)
        for key in self.Z.keys():
            eos.Z[key] = self.Z[key][:,iy,:].reshape(eos.shape)
        for key in self.qK.keys():
            eos.qK[key] = self.qK[key][:,iy,:].reshape(eos.shape)

        return eos

    def slice_at_t_idx(self, it):
        """
        Constructs a new table at a fixed temperature self.t[it]
        """
        eos = self.copy(copy_data=False)
        eos.t = np.array(eos.t[it], dtype=self.dtype).reshape((-1))
        eos.shape = (eos.nb.shape[0], eos.yq.shape[0], 1)

        for key in self.thermo.keys():
            eos.thermo[key] = self.thermo[key][:,:,it].reshape(eos.shape)
        for key in self.Y.keys():
            eos.Y[key] = self.Y[key][:,:,it].reshape(eos.shape)
        for key in self.A.keys():
            eos.A[key] = self.A[key][:,:,it].reshape(eos.shape)
        for key in self.Z.keys():
            eos.Z[key] = self.Z[key][:,:,it].reshape(eos.shape)
        for key in self.qK.keys():
            eos.qK[key] = self.qK[key][:,:,it].reshape(eos.shape)

        return eos

    def validate(self, check_cs2_min=False, check_cs2_max=True):
        """
        Mark invalid points in the table
        """
        self.valid[:] = True
        if check_cs2_min:
            self.valid = self.valid & (self.thermo["cs2"] > 0)
        if check_cs2_max:
            self.valid = self.valid & (self.thermo["cs2"] < 1)

    @staticmethod
    def _extend_copy(arr, n_nb, n_t):
        sn, sy, st = arr.shape
        new_shape = (sn+n_nb, sy, st+n_t)
        new = np.zeros(new_shape, dtype=arr.dtype)
        new[:sn, :sy, :st] = arr
        new[sn:, :sy, :st] = arr[-1][None, :]
        new[:sn, :sy, st:] = arr[:, :, -1][:, :, None]
        new[sn:, :sy, st:] = arr[-1, :, -1][None, :, None]
        return new

    def extend_table(self, n_nb, n_t):
        '''
        This adapts the logic in
        https://bitbucket.org/FreeTHC/thcextra/src/master/EOS_Thermal_Extable
        to extend the table to higher T and nb by n_nb and n_t points, respectively.
        Namely, all fields except pressure and internal energy are simply copied
        from the last point in the table.
        The internal energy and pressure are calculated via:

        eps = eps(rho_cap, T_cap, Y_e) + eps_th(T) + delta_eps(rho, Y_e)
        P = P(rho_cap, T_cap, Y_e) + Gamma_th (rho_cap, Y) rho eps_th

        with

        rho_cap = min(rho, rho_max)
        T_cap = min(T, T_max)
        eps_th = max((T - T_max)/m_B, 0)
        delta_eps = P(rho_max, T_min, Y_e) (1/rho_max - 1/rho)
        Gamma_th = (P(rho, T_max, Y_e) - P(rho, T_min, Y_e))/(rho(eps(rho, T_max, Y_e) - eps(rho, T_min, Y_e)))

        Note that the Gamma_th addition to the pressure is not in Extable but is
        probably necessary in athena because the temperature needs to be recoverable
        via root finding of the pressure so it has to always depend monotonically on T.
        '''
        dln = np.log10(self.nb[-1]) - np.log10(self.nb[-2])
        dlt = np.log10(self.t[-1]) - np.log10(self.t[-2])
        sn, sy, st = self.shape
        new_shape = (sn+n_nb, sy, st+n_t)

        self.shape = new_shape
        self.nb = np.concatenate((self.nb, self.nb[-1]*10**(np.arange(1, n_nb+1)*dln)))
        self.t = np.concatenate((self.t, self.t[-1]*10**(np.arange(1, n_t+1)*dlt)))
        for grp in (self.thermo, self.Y, self.A, self.Z, self.qK):
            for key, data in grp.items():
                grp[key] = self._extend_copy(data, n_nb, n_t)
        self.valid = self._extend_copy(self.valid, n_nb, n_t)

        # calculate new pressure and epsilon
        p = self.thermo['Q1']*self.nb[:, None, None]
        eps = self.thermo['Q7']
        rho = self.nb[:, None, None]*self.mn

        p_th = (p - p[..., 0, None])
        eps_th = (eps - eps[..., 0, None])
        eps_th[..., 0] = eps_th[..., 1] # prevent div by 0
        gthm1 = p_th/(eps_th*rho)
        gthm1[sn:, :, :st] = gthm1[sn-1, :, :st]
        gthm1[:, :, st:] = gthm1[..., st-1, None]

        deps = np.zeros(new_shape)
        deps[sn:] = p[sn-1, :, 0, None]*(1/rho[sn-1] - 1/rho[sn:])
        th_eps = np.zeros(new_shape)
        th_eps[..., st:] = (self.t[st:] - self.t[st-1])/self.mn
        th_p = gthm1 * th_eps * rho

        self.thermo["Q1"] += th_p/self.nb[:, None, None]
        self.thermo["Q7"] += th_eps + deps

    def _write_data(self, dfile, dtype):
        dfile.attrs['version'] = self.version
        dfile.attrs['git_hash'] = self.git_hash
        dfile.create_dataset("nb", dtype=dtype, data=self.nb,
            compression="gzip", compression_opts=9)
        dfile.create_dataset("t", dtype=dtype, data=self.t,
            compression="gzip", compression_opts=9)
        dfile.create_dataset("yq", dtype=dtype, data=self.yq,
            compression="gzip", compression_opts=9)
        dfile["nb"].attrs["desc"] = "baryon number density [fm^-3]"
        dfile["t"].attrs["desc"] = "temperature [MeV]"
        dfile["yq"].attrs["desc"] = "charge fraction"

        dfile.create_dataset("mn", dtype=dtype, data=self.mn)
        dfile.create_dataset("mp", dtype=dtype, data=self.mp)
        dfile["mn"].attrs["desc"] = "neutron mass [MeV]"
        dfile["mp"].attrs["desc"] = "proton mass [MeV]"

        for name, desc in self.md.thermo.values():
            dfile.create_dataset(name, dtype=dtype, data=self.thermo[name],
                compression="gzip", compression_opts=9)
            dfile[name].attrs["desc"] = desc

        for name, desc in self.md.pairs.values():
            key = "Y[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.Y[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

        for name, desc in self.md.quads.values():
            key = "Y[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.Y[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

            key = "A[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.A[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

            key = "Z[{}]".format(name)
            dfile.create_dataset(key, dtype=dtype, data=self.Z[name],
                compression="gzip", compression_opts=9)
            dfile[key].attrs["desc"] = desc

        for name, desc in self.md.micro.values():
            dfile.create_dataset(name, dtype=dtype, data=self.qK[name],
                compression="gzip", compression_opts=9)
            dfile[name].attrs["desc"] = desc

    def write_hdf5(self, fname, dtype=np.float64):
        """
        Writes the table as an HDF5 file
        """
        with h5py.File(fname, "w") as dfile:
            self._write_data(dfile, dtype)

    def add_coldslice(self, fname, dtype=np.float64):
        """
        Add a cold table to the HDF5 file
        """
        assert self.shape[1] == 1
        assert self.shape[2] == 1

        with h5py.File(fname, "a") as dfile:
            cs_grp = dfile.require_group("cold_slice")
            self._write_data(cs_grp, dtype)
            cs_grp["lorene_cut"] = self.lorene_cut

    def write_athtab(self, fname, dtype=np.float64, endian='native'):
        """
        Writes the table as a .athtab file for use in AthenaK
        """
        # Prepare the header
        endianness = '='
        endianstring = sys.byteorder
        if endian == 'big':
            endianness = '>'
            endianstring = 'big'
        elif endian == 'little':
            endianness = '<'
            endianstring = 'little'
        elif endian != 'native':
            raise RuntimeError(f'Unknown endianness {endianness}')
        fptype = dtype
        precision = 'double'
        fspec = 'd'
        if dtype == np.float32:
            precision = 'single'
            fspec = 'f'
        # Write header in ASCII
        with open(fname, "w") as f:
            # Generate metadata
            f.write(f"<metadatabegin>\n" \
                    f"version=1.0\n" \
                    f"endianness={endianstring}\n" \
                    f"precision={precision}\n" \
                    f"<metadataend>\n")
            # Print scalars
            f.write(f"<scalarsbegin>\n" \
                    f"mn={self.mn}\n" \
                    f"mp={self.mp}\n" \
                    f"<scalarsend>\n")
            # Prepare points
            # Note that because our fields will be indexed as (nb, yq, t), we must
            # write the points in this same order.
            f.write(f"<pointsbegin>\n" \
                    f"nb={len(self.nb)}\n" \
                    f"yq={len(self.yq)}\n" \
                    f"t={len(self.t)}\n" \
                    f"<pointsend>\n")
            # Prepare fields
            f.write(f"<fieldsbegin>\n")
            for name, desc in self.md.thermo.values():
                f.write(f"{name}\n")
            for name, desc in self.md.pairs.values():
                f.write(f"Y[{name}]\n")
            for name, desc in self.md.quads.values():
                f.write(f"Y[{name}]\n")
                f.write(f"A[{name}]\n")
                f.write(f"Z[{name}]\n")
            for name, desc in self.md.micro.values():
                f.write(f"{name}\n")
            f.write(f"<fieldsend>\n")
        # Now open the file in binary and write the data
        nn = len(self.nb)
        ny = len(self.yq)
        nt = len(self.t)
        npts = nn*ny*nt
        with open(fname, "ab") as f:
            f.write(struct.pack(f'{endianness}{nn}{fspec}', *self.nb))
            f.write(struct.pack(f'{endianness}{ny}{fspec}', *self.yq))
            f.write(struct.pack(f'{endianness}{nt}{fspec}', *self.t))
            for name, desc in self.md.thermo.values():
                f.write(struct.pack(f'{endianness}{npts}{fspec}',
                                    *self.thermo[name].flatten()))
            for name, desc in self.md.pairs.values():
                f.write(struct.pack(f'{endianness}{npts}{fspec}',
                                    *self.Y[name].flatten()))
            for name, desc in self.md.quads.values():
                f.write(struct.pack(f'{endianness}{npts}{fspec}',
                                    *self.Y[name].flatten()))
                f.write(struct.pack(f'{endianness}{npts}{fspec}',
                                    *self.A[name].flatten()))
                f.write(struct.pack(f'{endianness}{npts}{fspec}',
                                    *self.Z[name].flatten()))
            for name, desc in self.md.micro.values():
                f.write(struct.pack(f'{endianness}{npts}{fspec}',
                                    *self.qK[name].flatten()))

    def write_lorene(self, fname, subsample=1):
        """
        Export the table in LORENE format. This is only possible for 1D tables.
        """
        assert self.shape[1] == 1
        assert self.shape[2] == 1

        with open(fname, "w") as f:
            f.write("#\n#\n#\n#\n#\n%d\n#\n#\n#\n" % (len(self.nb) - self.lorene_cut))
            for ind, i in enumerate(range(self.lorene_cut, len(self.nb),subsample)):
                nb = self.nb[i]
                e  = Table.unit_dens*self.nb[i]*self.mn*(self.thermo["Q7"][i,0,0] + 1)
                p  = Table.unit_press*self.thermo["Q1"][i,0,0]*self.nb[i]
                f.write("%d %.15e %.15e %.15e\n" % (ind+1, nb, e, p))

    def write_rns(self, fname):
        """
        Export the table in RNS format. This is only possible for 1D tables.
        """
        assert self.shape[1] == 1
        assert self.shape[2] == 1

        icut = self.lorene_cut
        sed = self.thermo["Q7"][icut:,0,0]
        nb = self.nb[icut:]
        p = self.thermo["Q1"][icut:,0,0]*nb
        p -= p[0]
        ed = (1 + sed) * self.mn * nb
        h = sint.cumulative_trapezoid(1.0/(ed + p), p)

        nd_CGS = 1e39 * nb
        tmd_CGS = Table.unit_dens * ed
        p_CGS = Table.unit_press * p
        h_CGS = Table.unit_eps * h

        h_CGS[0] = 1 # Exact value = 0, but RNS seems to require that.
        p_CGS[0] = 1

        with open(fname, 'w') as f:
            f.write(f"{len(tmd_CGS):d} \n")
            for ed,p,h,n in zip(tmd_CGS, p_CGS, h_CGS, nd_CGS):
                f.write(f"{ed:.15e}  {p:.15e}  {h:.15e}  {n:.15e} \n")

    def write_txt(self, fname):
        """
        Export the table in TXT format. This is only possible for 1D tables.
        """
        assert self.shape[1] == 1
        assert self.shape[2] == 1

        with open(fname, "w") as f:
            f.write("# 1:nb 2:rho 3:press\n")
            for i in range(len(self.nb)):
                nb = self.nb[i]
                e  = self.nb[i]*self.mn*(self.thermo["Q7"][i,0,0] + 1)
                p  = self.thermo["Q1"][i,0,0]*self.nb[i]
                f.write("%.15e %.15e %.15e\n" % (nb, e, p))

    def add_Ye_from_yq(self):
        """
        Add Y["e"] = yq to table
        """
        assert not "e" in self.Y.keys()

        self.Y["e"] = self.yq[np.newaxis,:,np.newaxis]*np.ones(self.shape)

    def enfoce_positive_Q1(self,Q1_min=1e-16,P_min=None):
        """
        Enforce positivity of Q1.

        P_min is supplied then a floor is placed on the pressure, else the floor Q1_min is placed on Q1.
        """

        if P_min:
            P = self.thermo["Q1"]*self.nb[:,np.newaxis,np.newaxis]
            P = np.clip(P,P_min,None)
            self.thermo["Q1"] = P/self.nb[:,np.newaxis,np.newaxis]
        
        else:
            self.thermo["Q1"] = np.clip(self.thermo["Q1"], Q1_min, None)

    def write_number_fractions(self, fname):
        """
        Export an ASCII table with number fractions to complement the LORENE one.
        This is only possible for 1D tables.
        """
        assert self.shape[1] == 1
        assert self.shape[2] == 1

        with open(fname, "w") as f:
            keys = list(self.Y.keys())
            L    = len(self.Y[keys[0]])

            f.write("#\n#\n#\n#\n#\n%d" %L)
            for key in keys: f.write(" Y_%s" %key)
            f.write("\n#\n#\n#\n")

            for i in range(L):
                f.write("%d" %i)
                for key in self.Y.keys():
                    yi = self.Y[key][i]
                    f.write(" %.15e" %yi)
                f.write('\n')
