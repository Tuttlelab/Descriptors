#!/usr/bin/env python3
"""
adi_analysis.py

Top-level CLI script for Aggregate Dynamics Index (ADI).
Delegates calculation to the `descriptors.adi` module.
"""

from descriptors.adi import main

if __name__ == '__main__':
    main()