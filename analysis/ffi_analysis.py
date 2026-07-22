#!/usr/bin/env python3
"""
ffi_analysis.py

Top-level CLI script for Fiber Formation Index (FFI).
Delegates calculation to the `descriptors.ffi` module.
"""

from descriptors.ffi import main

if __name__ == '__main__':
    main()
