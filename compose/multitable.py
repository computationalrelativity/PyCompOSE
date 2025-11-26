
import sys
import struct
import numpy as np
from eos import Metadata, Table

try:
    from ._version import version
except ImportError:
    version = "unknown"

class MultiTable:
    def __init__(self, dtype=np.float64):
        """
        Initialize a MultiTable EOS object

        * dtype : data type
        """
        self.dtype = dtype
        
        self.table_3D_names = []
        self.table_2D_names = []

        self.tables_3D = {}
        self.tables_2D = {}

        self.version = version
        self.git_hash = version.split("+g")[-1]

        self.nscalars = 0

    def set_scalar_weights(self,nscalars,Y_weights,N_weights):
        assert np.issubdtype(Y_weights.dtype, np.integer), "Y_weights must have integer type, found: "+str(Y_weights.dtype)
        assert np.issubdtype(N_weights.dtype, np.integer), "N_weights must have integer type, found: "+str(N_weights.dtype)

        self.nscalars = nscalars

        ntables_3D = len(self.tables_3D)
        ntables_2D = len(self.tables_2D)

        if Y_weights.shape != (ntables_3D,nscalars+1):
            print("Y_weights does not match expected shape: ",(ntables_3D,nscalars+1))
        
        if N_weights.shape != (ntables_3D+ntables_2D,nscalars+1):
            print("N_weights does not match expected shape: ",(ntables_3D+ntables_2D,nscalars+1))

        self.Y_weights = Y_weights
        self.N_weights = N_weights

    def init_CompOSE_table(self, name, dims=3, metadata=Metadata()):
        new_table = Table(metadata=metadata, dtype=self.dtype)
        self.table_3D_names.append(name)

        if dims==3:
            self.tables_3D[name] = new_table
        else:
            assert dims==3, "Only 3D CompOSE tables are currently implemented for MultiTable"
        
        return
    
    def init_MuEOS_table(self, name, dims=2):
        new_table = MuEOSTable(dtype=self.dtype)
        self.table_2D_names.append(name)

        if dims==2:
            self.tables_2D[name] = new_table
        else:
            assert dims==2, "Only 2D MuEOS tables are currently implemented for MultiTable"
    
        return
    
    def calc_CompOSE_values(self, table):
        p = table.thermo["Q1"] * table.nb[:, np.newaxis, np.newaxis]
        s = table.thermo["Q2"] * table.nb[:, np.newaxis, np.newaxis] # This is entropy density (per volume) not entropy per baryon as in above cs2 calculation
        mub = table.thermo["Q3"]*table.mn
        muq = table.thermo["Q4"]*table.mn
        u = table.mn * (table.thermo["Q7"] + 1) * table.nb[:, np.newaxis, np.newaxis]

        if p.min() <= 0.0:
            p_ = p + 2 * max(sys.float_info.min, abs(p.min()))
        else:
            p_ = p

        if s.min() <= 0.0:
            s_ = s + 2 * max(sys.float_info.min, abs(s.min()))
        else:
            s_ = s

        dpdn = p_ * table.diff_wrt_nb(np.log(p_))
        dsdn = s_ * table.diff_wrt_nb(np.log(s_))

        if table.t.shape[0] > 1:
            dpdt = p_ * table.diff_wrt_t(np.log(p_))
            dsdt = s_ * table.diff_wrt_t(np.log(s_))
        else:
            dpdt = np.zeros(p.shape)
            dsdt = np.zeros(s.shape)

        table.pressure = p
        table.energy = u
        table.entropy = s
        table.mun = mub
        table.mup = mub + muq
        table.dpdn = dpdn
        table.dpdt = dpdt
        table.dsdn = dsdn
        table.dsdt = dsdt

        return

    def make_athtab_3D(self, name, fname):
        table = self.tables_3D[name]

        h_min = np.min(table.pressure + table.energy)

        data_dict = {"scalars" : [("dims", 3, "{:d}"),
                                  ("h_min", h_min, "{:21.16e}")], # (name, value, format string)
                     "points"  : [("ni", len(table.nb)),
                                  ("yi", len(table.yq)),
                                  ("t", len(table.t))], # (name, count)
                     "fields"  : ["pressure","energy","entropy",
                                  "dpdn","dpdt","dsdn","dsdt"], # (name)
                     "data"    : {"ni" : table.nb,
                                  "yi"  : table.yq,
                                  "t"  : table.t,
                                  "pressure" : table.pressure,
                                  "energy"   : table.energy,
                                  "entropy"  : table.entropy,
                                  "dpdn" : table.dpdn,
                                  "dpdt" : table.dpdt,
                                  "dsdn" : table.dsdn,
                                  "dsdt" : table.dsdt}} # values for points and fields

        self.write_athtab_file(fname, data_dict)
        return
    
    def make_athtab_2D(self, name, fname):
        table = self.tables_2D[name]

        h_min = np.min(table.pressure + table.energy)

        data_dict = {"scalars" : [("dims", 2, "{:d}"),
                                  ("h_min", h_min, "{:21.16e}")], # (name, value, format string)
                     "points"  : [("ni", len(table.n)),
                                  ("t", len(table.t))], # (name, count)
                     "fields"  : ["pressure","energy","entropy",
                                  "dpdn","dpdt","dsdn","dsdt"], # (name)
                     "data"    : {"ni" : table.n,
                                  "t"  : table.t,
                                  "pressure" : table.pressure,
                                  "energy"   : table.energy,
                                  "entropy"  : table.entropy,
                                  "dpdn" : table.dpdn,
                                  "dpdt" : table.dpdt,
                                  "dsdn" : table.dsdn,
                                  "dsdt" : table.dsdt}} # values for points and fields
 
        self.write_athtab_file(fname, data_dict)
        return
    
    def make_athtab_T_union(self, T_union, fname):
        data_dict = {"scalars" : [("dims",1,"{:d}")], # (name, value, format string)
                     "points"  : [("t",len(T_union))], # (name, count)
                     "fields"  : [], # (name)
                     "data"    : {"t" : T_union}} # values for points and fields

        self.write_athtab_file(fname, data_dict)
        return
    
    def write_athtab_file(self, fname, data_dict, dtype=np.float64, endian="native"):
        """
        Writes the table as a .athtab file for use in AthenaK
        """
        # Prepare the header
        endianness = "="
        endianstring = sys.byteorder
        if endian == "big":
            endianness = ">"
            endianstring = "big"
        elif endian == "little":
            endianness = "<"
            endianstring = "little"
        elif endian != "native":
            raise RuntimeError(f"Unknown endianness {endianness}")
        fptype = dtype
        precision = "double"
        fspec = "d"
        if dtype == np.float32:
            precision = "single"
            fspec = "f"
        # Write header in ASCII
        with open(fname, "w") as f:
            # Generate metadata
            f.write(
                f"<metadatabegin>\n"
                f"version=1.0\n"
                f"endianness={endianstring}\n"
                f"precision={precision}\n"
                f"<metadataend>\n"
            )

            # Write scalars
            scalar_list = data_dict["scalars"]
            scalar_str = "<scalarsbegin>\n"
            for idx, item in enumerate(scalar_list):
                name, value, format_str = item
                scalar_str += name + "=" + format_str.format(value) + "\n"
            scalar_str += "<scalarsend>\n"
            f.write(scalar_str)

            # Prepare points
            point_list = data_dict["points"]
            point_str = "<pointsbegin>\n"
            for idx, item in enumerate(point_list):
                name, value = item
                point_str += name + "=" + "{:d}".format(value) + "\n"
            point_str += "<pointsend>\n"
            f.write(point_str)

            # Prepare fields
            field_list = data_dict["fields"]
            field_str = "<fieldsbegin>\n"
            for idx, item in enumerate(field_list):
                name = item
                field_str += name + "\n"
            field_str += "<fieldsend>\n"
            f.write(field_str)

        with open(fname, "ab") as f:
            point_list = data_dict["points"]
            for points_name, data_size in point_list:
                data_current = data_dict["data"][points_name]
                f.write(struct.pack(f"{endianness}{data_size}{fspec}", *data_current))

            field_list = data_dict["fields"]
            for field_name in field_list:
                data_current = data_dict["data"][field_name]
                data_size = data_current.size
                f.write(struct.pack(f"{endianness}{data_size}{fspec}", *data_current.flatten()))
        
        return

    def write_athtab(self, output_dir, fnames={}, dtype=np.float64, endian="native"):
        n_ni_count = 0
        n_yi_count = 0
        n_T_count  = 0
        n_table_count = 0

        nvars = 7 # press, energy, entropy, dpdn, dpdt, dsdn, dsdt

        t_union = []

        min_n = -np.inf
        max_n =  np.inf
        min_T = -np.inf
        max_T =  np.inf
        min_Y =  np.full((self.nscalars,),-np.inf)
        max_Y =  np.full((self.nscalars,), np.inf)

        table_3D_output_fnames = []
        table_2D_output_fnames = []

        for table_idx, name in enumerate(self.table_3D_names):
            if name in fnames.keys():
                fname_current = fnames[name]
            else:
                fname_current = name + ".athtab"

            mb = self.tables_3D[name].mn
            ni_count_current = self.tables_3D[name].nb.shape[0]
            yi_count_current = self.tables_3D[name].yq.shape[0]
            T_count_current  = self.tables_3D[name].t.shape[0]
            for t in self.tables_3D[name].t:
                t_union.append(t)
            
            table_3D_output_fnames.append(fname_current)
            self.make_athtab_3D(name, output_dir + fname_current)
            
            min_n = max(min_n, self.tables_3D[name].nb[0])
            max_n = min(max_n, self.tables_3D[name].nb[-1])
            min_T = max(min_T, self.tables_3D[name].t[0])
            max_T = min(max_T, self.tables_3D[name].t[-1])
            for Y_idx in range(self.nscalars):
                min_Y[Y_idx] = max(self.tables_3D[name].yq[0]/self.Y_weights[table_idx,Y_idx+1],min_Y[Y_idx])
            for Y_idx in range(self.nscalars):
                max_Y[Y_idx] = min(self.tables_3D[name].yq[-1]/self.Y_weights[table_idx,Y_idx+1],max_Y[Y_idx])
            
            n_ni_count += ni_count_current
            n_yi_count += yi_count_current
            n_T_count  += T_count_current
            n_table_count += ni_count_current*yi_count_current*T_count_current*nvars

        n_tables_3D = len(table_3D_output_fnames)

        for name in self.table_2D_names:
            if name in fnames.keys():
                fname_current = fnames[name]
            else:
                fname_current = name + ".athtab"
    
            ni_count_current = self.tables_2D[name].n.shape[0]
            T_count_current  = self.tables_2D[name].t.shape[0]
            for t in self.tables_2D[name].t:
                t_union.append(t)
                
            table_2D_output_fnames.append(fname_current)
            self.make_athtab_2D(name, output_dir + fname_current)
            
            min_T = max(min_T, self.tables_2D[name].t[0])
            max_T = min(max_T, self.tables_2D[name].t[-1])
            
            n_ni_count += ni_count_current
            n_T_count  += T_count_current
            n_table_count += ni_count_current*T_count_current*nvars

        n_tables_2D = len(table_2D_output_fnames)

        t_union = np.sort(np.unique(np.array(t_union)))
        t_union = t_union[np.logical_and(t_union>=min_T,t_union<=max_T)]
        n_tunion = t_union.shape[0]

        if "T_union" in fnames.keys():
            fname_T_union = fnames["T_union"]
        else:
            fname_T_union = "T_union.athtab"

        self.make_athtab_T_union(t_union, output_dir + fname_T_union)

        if "header" in fnames.keys():
            fname_header = fnames["fnames"]
        else:
            fname_header = "header.multitable"

        with open(output_dir + fname_header, "w") as file_header:
            file_header.write("# MultiTable header file\n")
            
            file_header.write("# Number of each type of subtable\n")
            file_header.write("n_3D_tables = {:d}\n".format(n_tables_3D))
            file_header.write("n_2D_tables = {:d}\n".format(n_tables_2D))
            file_header.write("\n")
            
            file_header.write("# Number of species fractions to be used\n")
            file_header.write("n_species = {:d}\n".format(self.nscalars))
            file_header.write("\n")
            
            file_header.write("# Total storage required for subtables\n")
            file_header.write("n_ni    = {:d}\n".format(n_ni_count))
            file_header.write("n_yi    = {:d}\n".format(n_yi_count))
            file_header.write("n_T_tab = {:d}\n".format(n_T_count))
            file_header.write("n_T_union = {:d}\n".format(n_tunion))
            file_header.write("n_table = {:d}\n".format(n_table_count))
            file_header.write("\n")
            
            file_header.write("# Scalar values\n")
            file_header.write("mb = {:21.16e}\n".format(mb))
            file_header.write("min_n = {:21.16e}\n".format(min_n))
            file_header.write("max_n = {:21.16e}\n".format(max_n))
            file_header.write("min_T = {:21.16e}\n".format(min_T))
            file_header.write("max_T = {:21.16e}\n".format(max_T))
            for idx in range(self.nscalars):
                file_header.write("min_Y[{:d}] = {:21.16e}\n".format(idx,min_Y[idx]))
                file_header.write("max_Y[{:d}] = {:21.16e}\n".format(idx,max_Y[idx]))
            file_header.write("\n")
            
            file_header.write("# 3D table relative locations (n_3D_tables)\n")
            for table_idx, table_fname in enumerate(table_3D_output_fnames):
                file_header.write("table_3D_{:d} = ./{:s}\n".format(table_idx, table_fname))
            file_header.write("\n")
            
            file_header.write("# 2D table relative locations (n_2D_tables)\n")
            for table_idx, table_fname in enumerate(table_2D_output_fnames):
                file_header.write("table_2D_{:d} = ./{:s}\n".format(table_idx, table_fname))
            file_header.write("\n")
            
            file_header.write("# Union of temperatures table relative location\n")
            file_header.write("T_union = ./{:s}\n".format(fname_T_union))
            file_header.write("\n")
            
            file_header.write("# Y weights (n_3D_tables, 1+n_species)\n")
            for idx in range(n_tables_3D):
                for jdx in range(self.nscalars+1):
                    file_header.write(("wY_({:d},{:d}) = {:d}\n").format(idx,jdx,self.Y_weights[idx,jdx]))
            file_header.write("\n")
            
            file_header.write("# N weights (n_3D_tables+n_tables_2D, 1+n_species)\n")
            for idx in range(n_tables_3D + n_tables_2D):
                for jdx in range(self.nscalars+1):
                    file_header.write(("wN_({:d},{:d}) = {:d}\n").format(idx,jdx,self.N_weights[idx,jdx]))
            file_header.write("\n")

        return

class MuEOSTable:
    def __init__(self, dtype=np.float64):
        self.dtype=dtype

        self.n = np.empty(0)
        self.t = np.empty(0)
        self.shape = (self.n.shape[0], self.t.shape[0])

        self.version = version
        self.git_hash = version.split("+g")[-1]

    def read(self, path_to_table):
        """
        This reads from a lepton table produced by the MuEOS code.
        """

        # Data to be loaded from file
        dataset_dict = {
            0:  "nl",     # Number density of leptons
            1:  "nl_bar", # Number density of anti-leptons
            2:  "pl",     # Pressure of leptons
            3:  "pl_bar", # Pressure of anti-leptons
            4:  "el",     # Internal energy density of leptons
            5:  "el_bar", # Internal energy density of anti-leptons
            6:  "sl",     # Entropy density of leptons
            7:  "sl_bar", # Entropy density of anti-leptons
            8:  "mul",    # Chemical potential of leptons
            9:  "dpdn",   # Derivative of pressure wrt lepton number density
            10: "dsdn",   # Derivative of entropy density wrt lepton number density
            11: "dpdt",   # Derivative of pressure wrt temperature
            12: "dsdt",   # Derivative of entropy density wrt temperature
            }
        num_vars = 13
        
        # Load table header
        table_header = np.loadtxt(path_to_table, dtype=int, max_rows=2, ndmin=1)
        num_nl = table_header[0]
        num_t = table_header[1]

        log_nl = np.loadtxt(path_to_table,dtype=self.dtype,skiprows=2,max_rows=1,ndmin=1)
        log_t  = np.loadtxt(path_to_table,dtype=self.dtype,skiprows=3,max_rows=1,ndmin=1)

        assert(log_nl.shape[0]==num_nl)
        assert(log_t.shape[0]==num_t)

        self.n = 10.0**log_nl
        self.t = 10.0**log_t

        self.shape = (self.n.shape[0], self.t.shape[0])

        data_raw = np.loadtxt(path_to_table, dtype=self.dtype, skiprows=4, ndmin=2)

        assert data_raw.shape == (num_vars*num_nl,num_t)

        data_dict = {}
        for var_idx, name in dataset_dict.items():
            data_dict[name] = np.zeros(self.shape, dtype=self.dtype)
            data_dict[name][:,:] = data_raw[num_nl*var_idx:num_nl*(var_idx+1),:]
        
        # Pressure
        self.pressure = data_dict["pl"] + data_dict["pl_bar"]

        # Energy density
        self.energy = data_dict["el"] + data_dict["el_bar"]

        # Entropy density
        self.entropy = data_dict["sl"] + data_dict["sl_bar"]

        # Chemical potential
        self.mul = data_dict["mul"]

        # Derivative of pressure wrt lepton number density
        self.dpdn = data_dict["dpdn"]

        # Derivative of entropy density wrt lepton number density
        self.dsdn = data_dict["dsdn"]

        # Derivative of pressure wrt temperature
        self.dpdt = data_dict["dpdt"]

        # Derivative of entropy density wrt temperature
        self.dsdt = data_dict["dsdt"]

        return