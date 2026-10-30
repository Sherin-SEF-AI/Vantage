"""A fake sumolib (the part the CARLA co-simulation uses): checkBinary, net.readNet, miscutils.getFreeSocketPort."""
import os

from . import miscutils, net


def checkBinary(name, bindir=None):
    home = os.environ.get('SUMO_HOME', '')
    path = os.path.join(bindir or os.path.join(home, 'bin'), name)
    return path if os.path.isfile(path) else name
