# Reference bird's-eye stitcher

Two small programs that turn one frame per camera into the stitched bird's-eye view, using the remap tables in the
folder above this one (`bev_lut.txt`, `bev_lut.json`, `bev_lut.npz`, `<tag>_map.bin`, `<tag>_weight.bin`,
`body_mask.bin`). Use them to see the tables work on the vehicle, and as the reference a production pipeline
(NPP, VPI, CUDA or OpenGL) is checked against: both match Vantage's own stitcher to within a grey level.

## C++ (`bev_stitch.cpp`)

No dependencies; OpenMP when the compiler has it. The `vbev` namespace (`load`, `estimate_gains`, `stitch`) is the
part to embed in a capture loop; `main` reads binary PGM/PPM frames and writes a PPM.

```sh
g++ -O3 -std=c++17 -fopenmp bev_stitch.cpp -o bev_stitch      # or: cmake -B build && cmake --build build
./bev_stitch .. out.ppm F.ppm R.ppm ML.ppm MR.ppm QL.ppm QR.ppm   # frames in bev_lut.txt order
./bev_stitch .. out.ppm --frames captures/ --balance --bench 200  # captures/<tag>.ppm or .pgm; timing
```

## Python (`bev_stitch.py`)

numpy and OpenCV only. Stills, or live V4L2 / GStreamer / RTSP sources; `--cuda` remaps on the GPU when OpenCV was
built with CUDA (Jetson).

```sh
python3 bev_stitch.py --frames captures/ --out bev.png --balance
python3 bev_stitch.py --source F=/dev/video0 --source R=/dev/video1 --source ML=/dev/video2 \
    --source MR=/dev/video3 --source QL=/dev/video4 --source QR=/dev/video5 --balance --show
```

## What the tables mean

For output pixel `p` and camera `c`: `map_c[p]` is where to sample that camera, in sensor pixels with pixel centres at
`i + 0.5` (scale by `frame_width / sensor_width` for another resolution, subtract 0.5 for OpenCV-style sampling); -1
where the camera does not see `p`. `weight_c[p]` (0..255) is its share; the shares of all cameras sum to about 255.

    out[p] = sum_c weight_c[p] * gain_c * sample(frame_c, map_c[p]) / sum_c weight_c[p]

`--balance` estimates `gain_c` per camera and channel from the ground two cameras share (Brown and Lowe gain
compensation), so cameras that auto-exposed differently meet without a step in brightness. On video it re-estimates
every few frames and smooths the gains so the view does not flicker.
