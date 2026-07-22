#!/usr/bin/env python3
"""
centering.py

Top-level CLI script for PBC Centering & Trajectory Preprocessing.
Delegates calculation to the `descriptors.centering` module.
"""

from descriptors.centering import main

if __name__ == '__main__':
    main()