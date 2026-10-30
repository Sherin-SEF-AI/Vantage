"""A fake `sumo` package (eclipse-sumo's): SUMO_HOME is tests/fake_sumo/sumo_home, with a fake bin/sumo (a Python script
that holds a TraCI socket; the fake traci runs the simulation) and tools/randomTrips.py."""
import os

SUMO_HOME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sumo_home')
__version__ = '1.23.1'
