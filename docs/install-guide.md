# Install guide

Calibrate ▸ **Install guide…** (or `vantage --install-guide rig.vantage`, full screen) takes a camera rig fitted to a
car from its live streams to a measured, verified calibration and coverage. It works one stage at a time, and each
stage is closed by a test, not by someone's judgement. It was built for six GMSL2 cameras on an NVIDIA Jetson AGX Orin,
but works with any source Vantage opens (a V4L2 device, a GStreamer pipeline, RTSP).

Code: `vantage/core/install_guide.py` (the stages and the mathematics, no Qt), `vantage/ui/workspaces/guide_ws.py` (the
page), `vantage/assistant/install_coach.py` (Claude as a coach). Tests: `tests/test_install_guide.py`.

## The page

- **Left:** the eight stages (✓ when the stage's test passes), what to do in the current one, its figures, and the
  settings: the requirement (1σ per angle and per position axis), how precisely the boards are laid, and the corner
  noise.
- **Centre:** every camera live, with what the stage needs drawn over it (below).
- **Right:** the ground coverage of the measured rig and the coach.
- **Sources:** *Live* reads `project.streams`, the same as Calibrate ▸ Live rig. *Rehearse* renders every camera's view
  of the planned boards from a pose a little off its plan, to practise the guide (and test it) without a car.
- **Buttons:**
  - *Save as-built* adds a variant with the measured poses.
  - *Report…* writes everything as JSON.
  - *Export layout…* writes every board's place as a CSV, to lay them out with a tape.

## The stages

| Stage | What happens | Passes when |
|---|---|---|
| Connect | Every stream's rate, drops and capture-time spread (`core.rig_health`) | Every camera at 90 % of its frame rate or more |
| Identify | The installer covers one lens at a time; the stream that goes dark is that camera's | Every camera matched to a stream (a swapped cable shows as "swapped") |
| Lens | Each camera's unit calibration, else the library model | Every camera calibrated, or the library models accepted |
| Aim | Each camera's turn on its mount, from the car's own outline in its image | Every axis the outline can show within ±1° |
| Boards | The planned boards found in every camera (yellow: where expected; green: corners found) | Every camera has found its boards |
| Solve | Every pose from the boards in one fit, with its 1σ per axis | σ × safety within the requirement on every axis, and χ² of the residuals consistent with the corner noise |
| Coverage | The ground the measured rig sees, graded robust (green), uncertain (amber) or blind (red) | Robust coverage within 0.5 % of the plan's |
| Verify | Tests the solve did not use | The check board, the seams, the residuals and the coverage all pass |

**Plan boards** (Boards stage) makes the board plan:
1. It keeps the project's boards that some camera sees.
2. It adds, one at a time, the board that most reduces the cameras' excess over the requirement. Candidates are
   0.6 m and 0.9 m boards on a 0.5 m grid round the car, laid square to it.
3. It stops when every camera meets the requirement.
4. It reserves a check board that two cameras see.
5. For a camera no floor board can reach, it plans a board on a stand, shared with the solved camera that looks most
   the same way.

Planning takes about 10–20 s on an AGX Orin.

## The mathematics

- **Pose covariance.** Corners have independent Gaussian noise σ on each pixel coordinate. J is the Jacobian of the
  corner pixels with respect to each camera's rotation vector and position, and each floor board's x, y and yaw. Board
  placement is a Gaussian prior P.
  - The covariance of the least-squares estimate is Σ = (JᵀJ/σ² + P)⁻¹. This is the Cramér–Rao bound of the
    linearised problem.
  - A camera's 6×6 block is its pose covariance with every board marginalised (Schur complement). Boards share no
    parameters, so each one is eliminated with its own 3×3 inverse.
  - When the fit's χ² says the corners are noisier than σ, Σ is inflated by s² = χ²/dof.
  - **Checked independently:** this agrees with Calibrate ▸ Calibration observability's covariance, which is separate
    code, within 12 % on every axis (`test_joint_covariance_matches_observability`).
- **Chained cameras** (a board on a stand seen by a solved parent p and a child c).
  - Unknowns x = (board pose, child pose). The parent's error Σ_P propagates through the chain:
    Σ_x = H⁻¹ + G Σ_P Gᵀ, with H = J_xᵀJ_x/σ² and G = −H⁻¹ J_xᵀ A/σ².
  - **Checked:** against 80 Monte Carlo solves, every axis within 35 % (`test_chain_covariance_matches_monte_carlo`).
- **Robust coverage.** A ground point seen at a margin m pixels from the nearest image edge (sensor edge or image
  circle), with the pixel's spread along that edge's normal σₙ = √(nᵀ J Σ Jᵀ n):
  - It is robust when m ≥ 3σₙ.
  - The chance that it really lies outside the image past that edge is then at most Φ(−3) = 0.13 % (first order).
- **Aim.** The planned body outline, as rays, is turned until it best meets edges of the image running the same way.
  - The search is coarse-to-fine: every axis on a smoothed image, then the best few basins refined on the sharp image.
  - The score's curvature says which axes the outline pins. A straight outline (a bumper's top edge) cannot tell a pan
    from a roll; such axes are reported as not pinned and left to the boards.
  - **Checked on synthetic captures:** pinned axes within 0.22° (median 0.04°).
  - **Limit:** a mount moved by millimetres also moves close body edges, so the aim is good to about ±0.5° on a real
    car. That is good enough for mounting, never a calibration.
  - The correction is given from behind the camera, with the shim: shim = base × tan(angle), so 0.70 mm turns a 40 mm
    base by 1°.
- **Boards in images.** The image is resampled onto the board's own plane (with the planned, aimed or solved pose).
  - Every board then looks square-on to OpenCV's detector, however obliquely the fisheye sees it.
  - A corner found in that view maps back to the exact image pixel it was sampled from, so the pose used only has to
    put the board inside the view.
  - **Checked on synthetic captures:** corners within 0.09 px median of the truth, and 13 of 18 boards found where the
    pinhole-patch detector of the Installation check found 9.
  - The planner counts a board only when its squares are 12 px or more: every board below that was missed.
- **Ground error from a tilt.** A pitch error δ moves a ground point at distance d, seen from height h, by
  δ (h² + d²)/h, since d = h/tan α.
- **Verify.**
  - The check board's corners, seen from two cameras, meet the floor at points that must agree: Dᵀ Σ_D⁻¹ D against
    χ²(2) for each corner, at 99 %. This test does not depend on where the board lies.
  - The seams' 90th percentile within 20 mm.
  - χ² of the fit at 1 %.

What is proven and what is assumed:
- **Proven:** the formulas above hold for the linearised Gaussian model.
- **Checked by simulation:** the agreement figures quoted above.
- **Assumed, and tested on the car by the verify stage:**
  - a flat floor and flat, truly printed boards;
  - the lens models;
  - corner noise independent from corner to corner;
  - errors small enough to linearise.

## Cameras behind glass

A camera inside the body's shell (behind the windscreen or the rear window) looks out through the first surface each
ray crosses. Anything beyond that, such as the bonnet or the boot lid, still blocks it. Many body models have no
separate windscreen part, which is why the rule is "inside the shell", not "behind a glass part".

## The coach

With a Claude key (Preferences ▸ Assistant), Claude comments when:
- a stage passes or fails;
- a camera drops out;
- the installer has been stuck on a stage for 90 s.

It comments at most every 20 s, and also answers questions.

What it sees and does:
- It reads the guide's state, and can look at a camera's current frame, sent as a JPEG at most 1024 px wide. That is
  for what the numbers cannot tell: glare, a covered lens, a bent or misplaced board.
- It can ask what turning a camera would do to the coverage.
- It never computes a calibration.
- Every figure in its answer must be one of the guide's. A sentence with any other figure is removed before the
  installer sees it, and the page says so.
- It uses `claude-opus-5-5` with adaptive thinking: effort low for events, medium for questions.

## On a Jetson AGX Orin with GMSL2 cameras

1. Set each camera's stream in Calibrate ▸ Live rig, for example `v4l2src device=/dev/video0 io-mode=2 !
   video/x-raw,format=UYVY,width=1920,height=1536,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx !
   videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1 sync=false`. GMSL port n is `/dev/video(n-1)`.
2. Run `sudo nvpmodel -m 0 && sudo jetson_clocks` for the full CPU.
3. Run `vantage --install-guide rig.vantage`.

The guide's linear algebra runs single-threaded (`install_guide.single_blas`). On cores the camera pipelines keep busy,
OpenBLAS's worker threads spin-wait: one 36 × 36 inverse took 62 ms on six threads, against 0.04 ms on one.
