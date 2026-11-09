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
`--template <vehicle-library id>|tigor` (the old sedan, hatchback, suv, van and truck still work), `--self-test`, `--version`.
