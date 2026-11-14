"""Vantage's edge watchdog: a headless service for the car's computer (a Jetson Orin or any Linux x86 or ARM64 box)
that reads the rig's camera streams, checks their health and whether a camera has been knocked out of its pose, and
alerts locally (status page, MQTT, webhook, log file, a buzzer line). No Qt and no OpenGL: the standard library,
NumPy and OpenCV (opencv-python-headless is enough), plus the Vantage package's Qt-free core modules it reuses.

    python -m vantage.edge --config /etc/vantage/edge.json            # live
    python -m vantage.edge --config edge.json --dry-run drive/ --inject faults.json --report run.json

The configuration is exported from a project (vantage.core.edge_daemon.export_config, or the Edge watchdog page). The
watchdog only reads streams and raises alerts: it commands nothing in the vehicle.
"""
SCHEMA = 'vantage.edge_config/1'


def main(argv=None) -> int:
    from .daemon import main as _main
    return _main(argv)
