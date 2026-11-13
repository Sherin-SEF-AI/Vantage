# Installing Vantage

How to install Vantage on Windows, macOS and Linux (including NVIDIA Jetson), check a download, run it from source and build the downloads. The short version is in the [README](../README.md).

Download the file for your system from the [latest release](https://github.com/Sherin-SEF-AI/Vantage/releases/latest).
Each one carries its own Python and libraries, so there is nothing else to install.

| System | File | Needs |
|---|---|---|
| Windows | `Vantage-<version>-windows-x64-setup.exe` (installer) or `…-windows-x64.zip` | Windows 10 or 11, x64 |
| macOS | `Vantage-<version>-macos-arm64.zip` | macOS 14 or later, Apple silicon |
| Linux | `Vantage-<version>-linux-x86_64.tar.gz` | x86_64, glibc 2.35 or later (Ubuntu 22.04, Debian 12, Fedora 36 and newer) |
| Linux arm64 | `Vantage-<version>-linux-arm64.tar.gz` | arm64, glibc 2.35 or later: NVIDIA Jetson Orin with JetPack 6, Ubuntu 22.04 arm64 and newer |

**Windows.** Run the setup. It installs for the current user without administrator rights (into
`%LOCALAPPDATA%\Programs\Vantage`), adds Vantage and *Vantage self-test* to the Start menu, and opens `.vantage`
projects. The installer is not signed, so SmartScreen says "Windows protected your PC": choose **More info**, then
**Run anyway**. `Vantage.exe` is the window; `vantage-cli.exe` in the same folder is for batch mode and `--self-test`
from a terminal. The `.zip` is the same program without the installer: unzip it anywhere and run `Vantage.exe`.
Uninstall from *Settings ▸ Apps*.

**macOS.** Unzip the file and move `Vantage.app` to Applications. The app is not signed or notarised, so macOS refuses
the first start: open *System Settings ▸ Privacy & Security* and choose **Open Anyway** next to the message about
Vantage. On macOS 14 you can instead Control-click the app and choose **Open**. If macOS says the app is damaged, run
`xattr -dr com.apple.quarantine /Applications/Vantage.app` in Terminal. Batch mode and the self-test:
`/Applications/Vantage.app/Contents/MacOS/vantage-cli --self-test`.

**Linux.**

```bash
tar xf Vantage-*-linux-x86_64.tar.gz      # or …-linux-arm64.tar.gz on an arm64 machine such as a Jetson
cd Vantage-*-linux-*/
./install.sh            # per user: ~/.local/opt/vantage, a menu entry, the .vantage file type, ~/.local/bin/vantage
vantage --self-test     # checks the analysis, the exports, cameras, ROS 2 bags, Qt's plugins, OpenGL and every page
```

`~/.local/opt/vantage/uninstall.sh` removes it again (projects and settings are kept). Without installing, run
`./Vantage` from the extracted folder.

**On the vehicle (NVIDIA Jetson).** The arm64 bundle runs on a Jetson Orin with JetPack 6, so the live rig health, the
drive-log check and the as-built check run on the data-collection computer itself. Cameras open as V4L2 devices
(`/dev/video0`, as e-con's GMSL drivers expose them) or as GStreamer pipelines such as
`nvarguscamerasrc sensor-id=0 ! nvvidconv ! video/x-raw,format=BGRx ! appsink`: the bundle's OpenCV has no GStreamer,
so Vantage runs the pipeline in the system's `gst-launch-1.0` (JetPack's, with NVIDIA's plug-ins) and reads the frames
back. `vantage --self-test` says which of these work on the machine. JetPack 5 (Ubuntu 20.04) is too old for the
bundle: run from source there.

**From source on a Jetson AGX Orin with JetPack 7** (L4T R39, Ubuntu 24.04, Python 3.12, CUDA 13): one script sets
up a virtual environment that uses the GPU and the system's camera stack:

```bash
sudo apt install python3-venv python3-opencv python3-numpy python3-scipy gstreamer1.0-tools   # once
packaging/jetson/setup.sh             # .venv/, PyTorch with CUDA, ~/.local/bin/vantage, the app-menu entry
packaging/jetson/setup.sh --self-test # the same, then the self-test (about 5 minutes on an AGX Orin 64 GB)
```

- **OpenCV comes from apt, not PyPI.** Ubuntu's `python3-opencv` is built with GStreamer, so `nvarguscamerasrc` and
  `v4l2src` pipelines open in OpenCV directly (the self-test's cameras line says *GStreamer pipelines through
  OpenCV*). The venv sees the system's packages, and the script installs Vantage with `--no-deps` so that
  `opencv-python` (no GStreamer) never shadows it.
- **numpy and scipy come from PyPI, inside the venv.** Their wheels bring OpenBLAS. Apt's use whatever
  `libblas.so.3` points at, which on a fresh JetPack is the reference BLAS: a 2000 × 2000 matrix product took 5.5 s
  against 0.22 s with OpenBLAS (`sudo apt install libopenblas0-pthread` fixes the system's too). numpy stays below 2
  because apt's OpenCV is built against numpy 1.x.
- **BLAS threads.** On a Jetson, Vantage starts OpenBLAS (and OpenMP) with half the cores unless
  `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` are already set. The cores are shared with the camera pipelines, and a
  thread per core spin-waits on the busy ones: with six GMSL cameras streaming, a 1000 × 1000 SVD took 14.9 s on 12
  threads and 0.7 s on 6.
- **PyTorch on the GPU.** The script installs PyTorch's `cu130` aarch64 wheel, which brings its own CUDA runtime (no
  CUDA toolkit needed, only JetPack's driver). The wheel carries sm_80 code and names Orin's sm_87 unsupported, but
  its kernels run correctly there: the self-test's `gpu` line runs a convolution and a grid sample, forward and
  backward, on the GPU and checks them against the CPU (`PyTorch 2.14.1+cu130 on Orin (sm_87, 66 GB shared with the
  CPU): results match the CPU`). Vantage hides PyTorch's warning about Orin for that reason. `--no-torch` leaves
  PyTorch out.
- **Brotli comes from PyPI too.** Apt's `python3-brotli` 1.1 is too old for the Anthropic SDK's HTTP client, which
  then reports every successful Claude call as "Could not reach Anthropic's API" (`process() takes no keyword
  arguments`). The requirements install Brotli 1.2 or later in the venv.
- **Boards in the Installation check.** Apt's OpenCV 4.6 places every other corner of a small, oblique board 1–2 px
  off (a checkerboard pattern of errors), which pulled the joint solve 0.37° off for one camera of the sample. On a
  Jetson the check looks for each board first on the board's own plane, as the Install guide does (every square
  40 px and seen square-on), then in the usual pinhole patch: on the sample's synthetic captures the worst corner went
  from 2.3 px to 0.25 px, the rear camera found three boards instead of one, and every camera came within 0.02°.
  The same OpenCV takes 1.8–2.7 s for its classic detector on a board-free patch of sensor noise (a planned lidar
  board that is not there: 10 s a frame, against 0.15 s on a desktop), so on a Jetson that detector runs at half
  size, where it takes 0.01 s, and its corners are refined at full size.
- **Slower CPU.** The pages that run again by themselves after an edit (calibration observability, coverage,
  exposure, critical scenarios, rig perception) do so while their last run took less than a few seconds. An Orin's
  cores take about three times as long as the desktop's those limits were set on, so on a Jetson they are three
  times as long (`vantage.CPU_SLOWDOWN`); the observability study takes 3–6 s there against a 3 s desktop limit.
- **Memory.** The Orin's GPU shares the 64 GB with the CPU, so the GPU jobs' memory budgets (*gpu_mem_gb* in log
  replay and the foundation models) are slices of system RAM: leave room for the recorder and the window.
- **Power mode.** For the long jobs (campaigns, log replay, designed experiments), run the Orin in MAXN with its
  clocks fixed: `sudo nvpmodel -m 0 && sudo jetson_clocks`.
- **The reference stitcher's `--cuda`** needs an OpenCV built with CUDA, which apt's is not. Without one it remaps on
  the CPU and says so.

**Checking a download.** `SHA256SUMS.txt` in the release lists the SHA-256 of every file:

```bash
sha256sum -c SHA256SUMS.txt --ignore-missing                  # Linux: checks the files you downloaded
shasum -a 256 Vantage-*-macos-arm64.zip                         # macOS: compare with its line in SHA256SUMS.txt
```

On Windows, `Get-FileHash .\Vantage-<version>-windows-x64-setup.exe` in PowerShell prints the hash to compare.

`vantage --self-test` (on Windows the *Vantage self-test* shortcut) checks an installation without opening a window:
it opens the sample, runs an analysis, writes every export format and a ROS 2 bag to a temporary folder, loads Qt's
platform plugins, draws with OpenGL and with the software renderer, and builds every page. It ends with
`self-test passed`, or names what is missing.

**With Python 3.10 or later:**

```bash
python3 -m pip install .            # from this folder; add [step] for STEP import through gmsh,
                                    # [bags] for compressed ROS 2 bags (zstd, lz4), [keyring] for the assistant's key,
                                    # [claude] for the Claude provider (Anthropic's SDK: or pip install --user anthropic),
                                    # [torch] for gradient refinement and log replay (PyTorch; the downloads leave it out),
                                    # [terminal] for the Terminal panel (pyte)
vantage                             # the window (vantage-gui on Windows starts it without a console)
vantage --install-desktop           # Linux: Vantage in the app menu and dock with its icon, and .vantage files
vantage --desktop-shortcut          # Linux: the same, plus a Vantage launcher on the desktop
vantage project.vantage --analyse report.json
```

**Building the downloads.** Tag a version (`git tag v0.8.1 && git push origin v0.8.1`). The *Build installers*
workflow builds and smoke-tests all four bundles (Linux x86_64 and arm64, Windows, macOS). When all pass, it publishes
the GitHub release with `SHA256SUMS.txt` and the CHANGELOG entry as notes. Starting the workflow by hand from the Actions tab builds the
bundles as artifacts only, without a release. To build for the machine you are on, in a fresh virtual environment
(the bundle requirements take OpenCV without its GUI):

```bash
python3 -m venv build-env && . build-env/bin/activate
python -m pip install -r packaging/requirements-bundle.txt
python packaging/build.py --smoke   # dist/: the bundle, the archive and on Windows the installer (needs Inno Setup 6)
```

## Run from source

```bash
cd vantage
python3 -m pip install -r requirements.txt     # PyQt6, numpy, scipy, PyOpenGL, PyYAML, opencv-python
python3 run_vantage.py                          # start window
python3 run_vantage.py --sample                 # straight into the Tigor EV sample
python3 run_vantage.py car.glb                  # import a model
python3 run_vantage.py project.vantage          # open a project
```

Options: `--theme dark|light|system`, `--scale 1.15`, `--software-3d` (QPainter 3D view instead of OpenGL),
`--template <vehicle-library id>|tigor` (the old sedan, hatchback, suv, van and truck still work), `--install-guide`
(open straight on Calibrate ▸ Install guide, full screen; [docs/install-guide.md](install-guide.md)), `--self-test`, `--version`.
