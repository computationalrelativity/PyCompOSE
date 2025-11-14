import ctypes
import glob
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
libfile_glob = glob.glob(os.path.join(script_dir, "..", "NQTLib*.so"))

if len(libfile_glob) == 0:
    print("Could not find library file NQTLib*.so.")
    print(
        "Library can be build by running 'python setup.py build' in path/to/PyCompOSE/compose/"
    )
    assert len(libfile_glob) > 0
libfile = libfile_glob[0]
NQT_lib = ctypes.CDLL(libfile)

NQT_lib.NQT_log_LANL_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_exp_LANL_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_log_LANL_single.restype = ctypes.c_double
NQT_lib.NQT_exp_LANL_single.restype = ctypes.c_double

NQT_lib.NQT_exp_ldexp_O1_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_log_frexp_O1_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_exp_ldexp_O1_single.restype = ctypes.c_double
NQT_lib.NQT_log_frexp_O1_single.restype = ctypes.c_double

NQT_lib.NQT_exp_ldexp_O2_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_log_frexp_O2_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_exp_ldexp_O2_single.restype = ctypes.c_double
NQT_lib.NQT_log_frexp_O2_single.restype = ctypes.c_double

NQT_lib.NQT_exp_O2_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_log_O2_single.argtypes = [ctypes.c_double]
NQT_lib.NQT_exp_O2_single.restype = ctypes.c_double
NQT_lib.NQT_log_O2_single.restype = ctypes.c_double

NQT_lib.NQT_log_LANL_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_exp_LANL_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_log_LANL_array.restype = None
NQT_lib.NQT_exp_LANL_array.restype = None

NQT_lib.NQT_log_frexp_O1_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_exp_ldexp_O1_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_log_frexp_O1_array.restype = None
NQT_lib.NQT_exp_ldexp_O1_array.restype = None

NQT_lib.NQT_log_frexp_O2_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_exp_ldexp_O2_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_log_frexp_O2_array.restype = None
NQT_lib.NQT_exp_ldexp_O2_array.restype = None

NQT_lib.NQT_log_O2_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_exp_O2_array.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64),
    np.ctypeslib.ndpointer(dtype=np.float64),
    ctypes.c_long,
]
NQT_lib.NQT_log_O2_array.restype = None
NQT_lib.NQT_exp_O2_array.restype = None


def NQT_log2_O1(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_log_LANL_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_log_LANL_single(x)
        return result


def NQT_exp2_O1(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_exp_LANL_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_exp_LANL_single(x)
        return result


def NQT_log2_frexp_O1(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_log_frexp_O1_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_log_frexp_O1_single(x)
        return result


def NQT_exp2_ldexp_O1(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_exp_ldexp_O1_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_exp_ldexp_O1_single(x)
        return result


def NQT_log2_frexp_O2(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_log_frexp_O2_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_log_frexp_O2_single(x)
        return result


def NQT_exp2_ldexp_O2(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_exp_ldexp_O2_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_exp_ldexp_O2_single(x)
        return result


def NQT_log2_O2(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_log_O2_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_log_O2_single(x)
        return result


def NQT_exp2_O2(x):
    if hasattr(x, "__len__"):
        x_array = np.array(x, dtype=np.float64).flatten()
        f_array = np.zeros(x_array.shape, dtype=np.float64)
        NQT_lib.NQT_exp_O2_array(x_array, f_array, x_array.__len__())
        if hasattr(x, "shape"):
            f_array = f_array.reshape(x.shape)
        return f_array
    else:
        result = NQT_lib.NQT_exp_O2_single(x)
        return result


if __name__ == "__main__":
    import datetime
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.family"] = "STIXGeneral"
    plt.close("all")

    # samples = 1000001
    # xs = 10.0**np.linspace(-3,3,num=samples)

    # start = datetime.datetime.now()
    # NQT_log_xs_C0 = NQT_log2(xs)
    # NQT_xs_C0 = NQT_exp2(NQT_log_xs_C0)
    # end = datetime.datetime.now()

    # time_taken_seconds = (end - start).total_seconds()
    # samples_per_second = samples / time_taken_seconds
    # print("Samples per second: {:.6e}".format(samples_per_second))

    # plot_samples = 10001
    # subsample_fac = max(samples//plot_samples,1)

    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(xs[::subsample_fac], np.log2(xs[::subsample_fac]),c="k")
    # ax[0].plot(xs[::subsample_fac], NQT_log_xs_C0[::subsample_fac],c="C0")
    # ax[0].set_xscale("log")
    # ax[0].set_ylabel(r"$\mathrm{log}_\mathrm{i}(x)$")
    # ax[1].plot(xs[::subsample_fac], NQT_log_xs_C0[::subsample_fac]-np.log2(xs[::subsample_fac]),c="C0",ls="--")
    # ax[1].set_xscale("log")
    # ax[1].set_xlabel(r"$x$")
    # ax[1].set_ylabel(r"$\mathrm{log}_\mathrm{NQT}(x) - \mathrm{log}_{2}(x)$")

    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(xs[::subsample_fac], xs[::subsample_fac])
    # ax[0].plot(xs[::subsample_fac], NQT_xs_C0[::subsample_fac])
    # ax[0].set_xscale("log")
    # ax[0].set_yscale("log")
    # ax[0].set_ylabel("$f^\prime (f(x))$")
    # ax[1].plot(xs[::subsample_fac], (NQT_xs_C0[::subsample_fac]-xs[::subsample_fac])/xs[::subsample_fac],c="C0",lw=0.5)
    # ax[1].set_xscale("log")
    # ax[1].set_xlabel(r"$x$")
    # ax[1].set_ylabel(r"$\Delta x / x$")

    # fig.savefig("NQT_test.pdf")
    # fig.savefig("NQT_test.png",dpi=480)

    print("Samples per second:")

    samples = 100000000
    xs = np.random.rand(samples)

    start_LANL = datetime.datetime.now()
    lxs_LANL = NQT_log2_O1(xs)
    end_LANL = datetime.datetime.now()
    print(
        "LANL (Bithack O1): {:.6e}".format(
            samples / ((end_LANL - start_LANL).total_seconds())
        )
    )

    start_frexp_O1 = datetime.datetime.now()
    lxs_frexp_O1 = NQT_log2_frexp_O1(xs)
    end_frexp_O1 = datetime.datetime.now()
    print(
        "frexp O1: {:.6e}".format(
            samples / ((end_frexp_O1 - start_frexp_O1).total_seconds())
        )
    )

    start_bithack_O2 = datetime.datetime.now()
    lxs_bithack_O2 = NQT_log2_O2(xs)
    end_bithack_O2 = datetime.datetime.now()
    print(
        "Bithack O2: {:.6e}".format(
            samples / ((end_bithack_O2 - start_bithack_O2).total_seconds())
        )
    )

    start_frexp_O2 = datetime.datetime.now()
    lxs_frexp_O2 = NQT_log2_frexp_O2(xs)
    end_frexp_O2 = datetime.datetime.now()
    print(
        "frexp O2: {:.6e}".format(
            samples / ((end_frexp_O2 - start_frexp_O2).total_seconds())
        )
    )

    import scipy.interpolate as interp

    samples = 10000
    powers = 2
    x0 = 2.0 ** (-powers)
    x1 = 2.0**powers

    xs_log2 = np.linspace(np.log2(x0), np.log2(x1), num=samples, dtype=np.float64)
    xs_NQT_O1 = NQT_log2_frexp_O1(2**xs_log2)
    xs_NQT_O2 = NQT_log2_frexp_O2(2**xs_log2)

    ys_exp2 = 2 ** np.linspace(np.log2(x0), np.log2(x1), num=samples, dtype=np.float64)
    ys_NQT_O1 = NQT_exp2_ldexp_O1(np.log2(ys_exp2))
    ys_NQT_O2 = NQT_exp2_ldexp_O2(np.log2(ys_exp2))

    fig0, ax0 = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax0[0].plot(2**xs_log2, xs_log2, lw=1.0, ls="-", c="k", label=r"$\log_2$")
    ax0[0].plot(
        2**xs_log2, xs_NQT_O1, lw=1.0, ls="-", c="C0", label=r"$\log_{\mathrm{NQT}}$"
    )
    ax0[0].set_xscale("log")
    ax0[0].legend()
    ax0[0].set_ylabel(r"$\log_{\mathrm{i}}(x)$")

    ax0[1].plot(
        2**xs_log2,
        xs_NQT_O1 - xs_log2,
        lw=1.0,
        ls="-",
        c="C0",
        label=r"$\log_{\mathrm{NQT}}$",
    )
    ax0[1].set_ylabel(r"$\Delta \log_{\mathrm{i}}(x)$")
    ax0[1].set_xlabel(r"$x$")

    fig0.savefig("test_log_NQT.pdf")

    fig0, ax0 = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax0[0].plot(
        2**xs_log2, xs_log2, lw=0.5, ls="-", c="k", label=r"$\mathcal{O}(\infty)$"
    )
    ax0[0].plot(
        2**xs_log2, xs_NQT_O1, lw=0.5, ls=":", c="C0", label=r"$\mathcal{O}(1)$"
    )
    ax0[0].plot(
        2**xs_log2, xs_NQT_O2, lw=0.5, ls="--", c="C1", label=r"$\mathcal{O}(2)$"
    )
    ax0[0].set_xscale("log")
    ax0[0].legend()
    ax0[0].set_ylabel(r"$\log(x)$")

    ax0[1].plot(
        2**xs_log2,
        xs_NQT_O1 - xs_log2,
        lw=0.5,
        ls=":",
        c="C0",
        label=r"$\mathcal{O}(1)$",
    )
    ax0[1].plot(
        2**xs_log2,
        xs_NQT_O2 - xs_log2,
        lw=0.5,
        ls="--",
        c="C1",
        label=r"$\mathcal{O}(2)$",
    )
    ax0[1].set_ylabel(r"$\Delta \log(x)$")
    ax0[1].set_xlabel(r"$x$")

    fig0.savefig("test_log_NQT_2.pdf")

    fig1, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1[0].plot(
        np.log2(ys_exp2), ys_exp2, lw=0.5, ls="-", c="k", label=r"$\mathcal{O}(\infty)$"
    )
    ax1[0].plot(
        np.log2(ys_exp2), ys_NQT_O1, lw=0.5, ls=":", c="C0", label=r"$\mathcal{O}(1)$"
    )
    ax1[0].plot(
        np.log2(ys_exp2), ys_NQT_O2, lw=0.5, ls="--", c="C1", label=r"$\mathcal{O}(2)$"
    )
    ax1[0].set_yscale("log")
    ax1[0].legend()
    ax1[0].set_ylabel(r"$\exp(x)$")

    ax1[1].plot(
        np.log2(ys_exp2),
        ys_NQT_O1 / ys_exp2,
        lw=0.5,
        ls=":",
        c="C0",
        label=r"$\mathcal{O}(1)$",
    )
    ax1[1].plot(
        np.log2(ys_exp2),
        ys_NQT_O2 / ys_exp2,
        lw=0.5,
        ls="--",
        c="C1",
        label=r"$\mathcal{O}(2)$",
    )
    ax1[1].set_ylabel(r"$\Delta \exp(x)$")
    ax1[1].set_xlabel(r"$x$")

    fig1.savefig("test_exp.pdf")
