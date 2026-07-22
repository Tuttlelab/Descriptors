#!/usr/bin/env python3
"""
shape_tracker.py

Top-level CLI script for Integrative Shape Tracking.
Delegates calculation to the `descriptors.tracker` module.
"""

from descriptors.tracker import main

if __name__ == '__main__':
    main()