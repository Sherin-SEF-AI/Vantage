__version__ = "0.8.1"

import os as _os
import warnings as _warnings

# On a Jetson the CPU cores are shared with the camera pipelines (GStreamer, nvvidconv, the recorder). OpenBLAS with a
# thread per core then spin-waits on busy cores: on an AGX Orin with six GMSL cameras streaming, a 1000 x 1000 SVD took
# 14.9 s with 12 threads against 0.7 s with 4-6. Half the cores, unless the environment already says otherwise; set
# before anything imports numpy.
if _os.path.exists('/etc/nv_tegra_release'):
    _n = str(max(1, (_os.cpu_count() or 2) // 2))
    for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS'):
        _os.environ.setdefault(_k, _n)

# PyTorch's CUDA wheels for aarch64 (cu130) carry sm_80 code and call a Jetson Orin's sm_87 unsupported on first use.
# On an AGX Orin (JetPack 7) the kernels run and match the CPU (`vantage --self-test` checks it on the machine: the gpu
# line), so the warning only alarms; any other GPU's warning is kept.
_warnings.filterwarnings('ignore', message=r'Found GPU\d+ Orin which is of compute capability', category=UserWarning)
