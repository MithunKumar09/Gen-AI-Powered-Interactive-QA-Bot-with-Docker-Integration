"""Provider-agnostic RAG core.

Pure logic: no Flask, no HTTP, no import-time network or provider construction.
The Flask layer in ``myapp`` is a thin adapter over :mod:`rag_core.pipeline`.
"""
