"""Simulator bridges. vantage.sim.carla_bridge runs a rig exported by core.simrig (sim_rig.json + sim_remap.npz) in the
CARLA simulator and records it as a nuScenes folder and a ROS 2 bag in core.dataset's layout.

This package runs in CARLA's own Python environment: it imports numpy, OpenCV (cv2), carla and, from Vantage, only
core.rosbag and core.mathutil (numpy and the standard library), never PyQt or the rest of Vantage's dependencies.
"""
