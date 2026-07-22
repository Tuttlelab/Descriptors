#!/usr/bin/env python3
"""
tfi_analysis.py

Top-level CLI script for Tube Formation Index (TFI).
Delegates calculation to the `descriptors.tfi` module.
"""

from descriptors.tfi import main

if __name__ == '__main__':
    main()
