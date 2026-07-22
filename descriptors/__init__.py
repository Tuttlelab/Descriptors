"""
Shape Descriptor Toolkit
========================

A Python toolkit for calculating shape descriptors and tracking morphological evolution
in peptide self-assembly simulations.

Published in Faraday Discussions (2025):
"Shape Descriptors for Peptide Self-Assembly"
https://pubs.rsc.org/en/content/articlelanding/2025/fd/d4fd00201f

Modules:
--------
- adi: Aggregate Dynamics Index
- sfi: Sheet Formation Index
- vfi: Vesicle Formation Index
- tfi: Tube Formation Index
- ffi: Fiber Formation Index
- tracker: Multi-Descriptor Shape Tracker
- centering: PBC Centering and Cluster Preprocessing
- utils: Shared I/O and MDAnalysis Utilities
"""

__version__ = "0.2.0"
__author__ = "Raj Kumar Rajaram Baskaran"
__citation__ = "Faraday Discussions, 2025, DOI: 10.1039/D4FD00201F"

from descriptors.adi import calculate_adi
from descriptors.sfi import calculate_sfi
from descriptors.vfi import calculate_vfi
from descriptors.tfi import calculate_tfi
from descriptors.ffi import calculate_ffi
from descriptors.tracker import track_shapes

__all__ = [
    "calculate_adi",
    "calculate_sfi",
    "calculate_vfi",
    "calculate_tfi",
    "calculate_ffi",
    "track_shapes",
]
