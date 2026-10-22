"""The design assistant: a chat model (Groq, OpenAI-compatible API) that answers questions about the open project
through tools that read Vantage's own analysis, and proposes camera changes the user applies (or not) as one undo step.

No Qt in this package: the UI drives it from a worker thread (ui/assistant.py)."""
