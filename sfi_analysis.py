#!/usr/bin/env python3
"""
sfi_analysis.py

Top-level CLI script for Sheet Formation Index (SFI).
Delegates calculation to the `descriptors.sfi` module.
"""

from descriptors.sfi import main

if __name__ == '__main__':
    main()