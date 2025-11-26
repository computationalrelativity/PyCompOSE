import os
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
PYCOMPOSEDIR = os.path.join(SCRIPTDIR, os.pardir)
assert os.path.exists(os.path.join(PYCOMPOSEDIR,"compose"))

import sys
sys.path.append(PYCOMPOSEDIR)
from compose.multitable import MultiTable

import numpy as np

multitable_eos = MultiTable()

tables_3D = ["baryons"]
tables_2D = ["electrons"]

baryon_table_dir = "/path/to/eos/dir/"
electron_table_dir = "/path/to/lepton/dir"
electron_table_fname = "eos_electrons_table.txt"
output_dir = "./"

tables_3D_locations = {tables_3D[0] : baryon_table_dir}
tables_2D_locations = {tables_2D[0] : electron_table_dir + electron_table_fname}

multitable_eos.init_CompOSE_table(tables_3D[0])
multitable_eos.init_MuEOS_table(tables_2D[0])


multitable_eos.tables_3D[tables_3D[0]].read(tables_3D_locations[tables_3D[0]])
multitable_eos.tables_2D[tables_2D[0]].read(tables_2D_locations[tables_2D[0]])

multitable_eos.calc_CompOSE_values(multitable_eos.tables_3D[tables_3D[0]])

nscalars = 1
Y_weights = np.array([[0,1]],dtype=int)
N_weights = np.array([[1,0],
                      [0,1]],dtype=int)

multitable_eos.set_scalar_weights(nscalars, Y_weights, N_weights)

multitable_eos.write_athtab(output_dir)