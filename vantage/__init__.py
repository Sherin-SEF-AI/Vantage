__version__ = "0.9.0"

import os as _os
import warnings as _warnings

# On a Jetson the CPU cores are shared with the camera pipelines (GStreamer, nvvidconv, the recorder). OpenBLAS with a
# thread per core then spin-waits on busy cores: on an AGX Orin with six GMSL cameras streaming, a 1000 x 1000 SVD took
# 14.9 s with 12 threads against 0.7 s with 4-6. Half the cores, unless the environment already says otherwise; set
# before anything imports numpy.
# True only on an NVIDIA Jetson (L4T). Everything Jetson-specific in Vantage is behind it, so other systems run
# unchanged.
JETSON = _os.path.exists('/etc/nv_tegra_release')

# How much longer a computation takes here than on the desktop the pages' "re-run by itself after an edit while the
# last run took less than N s" limits were set on. An AGX Orin's cores take about 3 times as long: the calibration
# observability study took 2 s warm and 3.7 s cold on its own, and 6.5 s beside the 3D view, against its 3 s limit.
CPU_SLOWDOWN = 3.0 if JETSON else 1.0

if JETSON:
    _n = str(max(1, (_os.cpu_count() or 2) // 2))
    for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        _os.environ.setdefault(_k, _n)

# On Windows, numpy's OpenBLAS with its own worker threads crashed the process (access violation, 0xC0000005) when
# Vantage's studies called it from several threads at once: the 0.9.0 bundle's self-test died in a different study on
# each run, and passed with one OpenBLAS thread. Vantage's parallelism is its study threads, and its matrices are
# small (3 x 3 rotations applied to many rays), so one BLAS thread each costs little. Unless the environment says
# otherwise; set before anything imports numpy.
if _os.name == 'nt':
    _os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

# PyTorch's CUDA wheels for aarch64 (cu130) carry sm_80 code and call a Jetson Orin's sm_87 unsupported on first use.
# On an AGX Orin (JetPack 7) the kernels run and match the CPU (`vantage --self-test` checks it on the machine: the gpu
# line), so the warning only alarms; any other GPU's warning is kept.
if JETSON:
    _warnings.filterwarnings('ignore', message=r'Found GPU\d+ Orin which is of compute capability',
                             category=UserWarning)
