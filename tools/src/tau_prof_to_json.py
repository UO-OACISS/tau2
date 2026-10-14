#!/usr/bin/env python3
"""Deprecated compatibility entry point; use :mod:`tau_prof2json`."""

import sys

from tau_prof2json import main


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except (OSError, ValueError) as error:
        print("tau_prof_to_json.py: {}".format(error), file=sys.stderr)
        sys.exit(2)
