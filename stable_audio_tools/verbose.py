"""Shared verbose flag for user-facing prints.

Defaults off. Toggle with set_verbose(True) or by setting SAT_VERBOSE=1.
"""
import os

VERBOSE = os.environ.get("SAT_VERBOSE", "0") == "1"


def set_verbose(v: bool):
    global VERBOSE
    VERBOSE = bool(v)


def vprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)
