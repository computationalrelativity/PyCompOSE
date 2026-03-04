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

# This example shows how to import/export CompOSE tables using the NQT logs
#
# Instructions
# - Create a folder "SFHo" at the same location as this script
# - Download https://compose.obspm.fr/eos/34 and place it in a folder "SFHo"
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
eos.read(os.path.join(SCRIPTDIR, "SFHo"))

# %%
eos.compute_cs2(floor=1e-6)
eos.compute_abar()
eos.validate()
# Remove the highest temperature point
eos.restrict_idx(it1=-1)
eos.shrink_to_valid_nb()

# %%
eos.write_hdf5(os.path.join(SCRIPTDIR, "SFHo", "SFHo.h5"))
eos.write_athtab(os.path.join(SCRIPTDIR, "SFHo", "SFHo.athtab"))

eos_NQT = eos.make_NQT_version()
eos_NQT.write_hdf5(os.path.join(SCRIPTDIR, "SFHo", "SFHo_NQT.h5"))
eos_NQT.write_athtab(os.path.join(SCRIPTDIR, "SFHo", "SFHo_NQT.athtab"))