#!/usr/bin/env python3
"""
vfi_analysis.py

Top-level CLI script for Vesicle Formation Index (VFI).
Delegates calculation to the `descriptors.vfi` module.
"""

from descriptors.vfi import main

if __name__ == '__main__':
    main()
