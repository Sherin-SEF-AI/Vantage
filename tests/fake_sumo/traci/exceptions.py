class TraCIException(Exception):
    """A command SUMO refused (the run goes on)."""


class FatalTraCIError(Exception):
    """The connection to SUMO is gone."""
