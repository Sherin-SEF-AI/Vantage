"""The lab server: a small self-hosted registry where a team (or several) publish Vantage projects as immutable,
digest-addressed versions, compare them, pool their fleets' calibration records into one fleet twin and pool their
perception records so the learned rig model trains on everyone's runs.

- store.py: the storage (files on disk, content-addressed, plus an SQLite index; a backup is the folder).
- server.py: the REST API on Python's standard library HTTP server (`vantage --lab-server ROOT`, or
  `python -m vantage.lab serve --root ROOT`), token auth, per-team access.
- client.py: publish, pull, sync, compare, the pooled fleet twin and the records, for the app and batch.

What leaves the user's machine is exactly the project files they publish (core.project_dir.publishable: the project
folder's files; the per-user view file is not sent). Stream passwords and API keys are never in project files
(core.credentials, assistant.secrets); the lab token is kept in the system keyring or a private file, never in a
project. TLS: run the server behind a reverse proxy that terminates HTTPS (see server.py's docstring)."""

API_VERSION = 1
SERVICE = 'vantage-lab'
DEFAULT_PORT = 8765
ROLES = ('admin', 'member', 'reader')
