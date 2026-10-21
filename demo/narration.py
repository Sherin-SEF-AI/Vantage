"""Narration for the Vantage demo tour.

Each scene is a list of lines; a line is (caption, spoken). The spoken text is what the voice reads: acronyms and
part numbers are spelled the way they are said. The tour (tour.py) starts each line at a point in the action and the
mixer (mix.py) places the recorded line there.
"""

SCENES = [
    ('intro', 'Vantage', [
        ('Vantage is a desktop tool for planning, analysing and checking surround-view camera systems on vehicles.',
         'Vantage is a desktop tool for planning, analysing, and checking surround-view camera systems on vehicles.'),
        ("In the next few minutes we'll place six cameras on a Tata Tigor EV, prove their coverage, and verify the "
         "installation as built.",
         "In the next few minutes, we'll place six cameras on a Tata Tee-gor E V, prove their coverage, and verify the "
         "installation as built."),
    ]),
    ('start', 'Projects', [
        ('Projects open from the start window: recent files, templates, or a vehicle model to import.',
         'Projects open from the start window: recent files, templates, or a vehicle model to import.'),
        ('We open the Tigor sample. Its body was measured from a nineteen-frame studio photo spin, and it carries six '
         'e-con Systems STURDeCAM31 cameras with fisheye lenses.',
         'We open the Tee-gor sample. Its body was measured from a nineteen-frame studio photo spin, and it carries six '
         'e-con Systems Sturdy-Cam thirty-one cameras, with fisheye lenses.'),
        ('The outliner lists the vehicle, cameras, compute unit, calibration targets and analysis; the inspector edits '
         'whatever is selected.',
         'The outliner lists the vehicle, the cameras, the compute unit, calibration targets, and the analysis. The '
         'inspector edits whatever is selected.'),
    ]),
    ('coverage', 'Live coverage analysis', [
        ("Coverage is computed live on a ground grid around the car. Each camera's footprint has its own colour, and "
         "body occlusion is exact for every camera.",
         "Coverage is computed live, on a ground grid around the car. Each camera's footprint has its own colour, and "
         "body occlusion is exact for every camera."),
        ('Switch the heat map to see pixel density, or how many cameras see each patch of ground.',
         'Switch the heat map to see pixel density, or how many cameras see each patch of ground.'),
        ('The coverage panel sums it up: blind area in the region of interest, minimum density, and ground seen twice.',
         'The coverage panel sums it up: the blind area in the region of interest, the minimum density, and the ground '
         'that is seen twice.'),
    ]),
    ('edit', 'Placing cameras', [
        ("Select a camera to edit it. Here's the left mirror camera.",
         "Select a camera to edit it. Here's the left mirror camera."),
        ('Change its pitch, and the blind area and density update as you move.',
         'Change its pitch, and the blind area and density update as you move.'),
        ('Four views at once: perspective, top, the simulated fisheye image through the camera, and the stitched '
         "bird's-eye view.",
         "Four views at once. Perspective, top, the simulated fisheye image through the camera, and the stitched "
         "bird's-eye view."),
    ]),
    ('seams', "Bird's-eye view and seams", [
        ("The bird's-eye view is stitched from all six cameras. Seams are placed automatically, and you can drag any "
         "seam to where the image is best.",
         "The bird's-eye view is stitched from all six cameras. Seams are placed automatically, and you can drag any "
         "seam to where the image is best."),
        ('Variants keep alternative layouts, and the compare page puts two of them side by side with a difference map.',
         'Variants keep alternative layouts, and the compare page puts two of them side by side, with a difference map.'),
    ]),
    ('tolerance', 'Tolerance study', [
        ('Brackets are never perfect. The tolerance study perturbs every camera thousands of times within the '
         'tolerances you set.',
         'Brackets are never perfect. The tolerance study perturbs every camera thousands of times, within the '
         'tolerances you set.'),
        ('It shows the spread of blind area, which camera and axis matter most, and the worst cases, pinned in 3D.',
         'It shows the spread of blind area, which camera and axis matter most, and the worst cases, pinned in 3 D.'),
    ]),
    ('optimiser', 'Placement optimiser', [
        ('The optimiser searches for better placements, scored under the same tolerances, and returns the best '
         'trade-offs between coverage, camera count and cable length.',
         'The optimiser searches for better placements, scored under the same tolerances, and returns the best '
         'trade-offs, between coverage, camera count, and cable length.'),
    ]),
    ('targets', 'Calibration targets', [
        ('Calibration boards are planned on the floor, and each one is checked against the cameras that must see it.',
         'Calibration boards are planned on the floor, and each one is checked against the cameras that must see it.'),
    ]),
    ('install', 'Installation check', [
        ('Once the cameras are fitted, the installation check closes the loop.',
         'Once the cameras are fitted, the installation check closes the loop.'),
        ('Each image comes from a file, a live stream, or, as here, a synthetic capture with a deliberate mounting '
         'error of up to one and a half degrees.',
         'Each image comes from a file, a live stream, or, as here, a synthetic capture, with a deliberate mounting '
         'error of up to one and a half degrees.'),
        ("The floor boards are found in every fisheye image, and each camera's installed pose is solved and compared "
         "with the plan, to a fraction of a degree and a few millimetres.",
         "The floor boards are found in every fisheye image, and each camera's installed pose is solved, and compared "
         "with the plan, to a fraction of a degree, and a few millimetres."),
        ('Stitched with the planned poses, the boards break apart at the seams. With the measured poses, they line up.',
         'Stitched with the planned poses, the boards break apart at the seams. With the measured poses, they line up.'),
        ('One click keeps the measured poses as an as-built variant, and the remap tables go straight to the ECU.',
         'One click keeps the measured poses as an as-built variant, and the remap tables go straight to the E C U.'),
    ]),
    ('loads', 'Load cases and link budget', [
        ('Load cases show what passengers and luggage do to the stitched view: with a full load, the seams ghost by '
         'twenty-three centimetres.',
         'Load cases show what passengers and luggage do to the stitched view. With a full load, the seams ghost by '
         'twenty-three centimetres.'),
        ('The compute inspector checks the bandwidth of every GMSL2 deserializer on the Jetson.',
         'The compute inspector checks the bandwidth of every G M S L 2 deserializer on the Jetson.'),
    ]),
    ('photo', 'Photo-matched view', [
        ('Reference photos are calibrated to the model, so you can look through any photo and see the body line up '
         'with the real car.',
         'Reference photos are calibrated to the model, so you can look through any photo, and see the body line up '
         'with the real car.'),
    ]),
    ('export', 'Export', [
        ('Export everything the next team needs: mount sheet and report, extrinsics, OpenCV calibration, a ROS 2 URDF, '
         "the Jetson camera config, bird's-eye remap tables, and mount points for CAD.",
         "Export everything the next team needs. The mount sheet and report, extrinsics, Open C V calibration, a Ross "
         "two U R D F, the Jetson camera config, bird's-eye remap tables, and mount points for CAD."),
        ('The analysis, the exports and the installation check also run from the command line, ready for the '
         'production line.',
         'The analysis, the exports, and the installation check also run from the command line, ready for the '
         'production line.'),
    ]),
    ('outro', 'Vantage', [
        ('Vantage. Plan it, prove it, and check it as built.',
         'Vantage. Plan it, prove it, and check it, as built.'),
    ]),
]


def lines():
    """[(scene id, line index, caption, spoken)]"""
    return [(sid, i, cap, spk) for sid, _, ls in SCENES for i, (cap, spk) in enumerate(ls)]
