"""A stand-in for CARLA's sumo_integration package (CARLA_0.9.16/Co-Simulation/Sumo/sumo_integration) for tests without
CARLA: the same classes, methods and conversions (BridgeHelper, SumoSimulation, CarlaSimulation, constants), shortened,
over the fake traci (tests/fake_sumo/traci) and whatever `carla` module is imported (tests/fake_carla in tests).
tests/test_sumo.py also runs the bridge against CARLA's real package when a CARLA folder is found."""
