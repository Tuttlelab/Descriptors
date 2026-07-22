

import os
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s %(message)s')
import MDAnalysis as mda
from descriptors.core.adi import calculate_adi

def test_calculate_adi_ff():
	# Paths to test data
	gro = "/users/mib23220/Documents/Descriptors/centered_files/high_ap/FF/" + [f for f in os.listdir("/users/mib23220/Documents/Descriptors/centered_files/high_ap/FF/") if f.endswith('.gro')][0]
	xtc = "/users/mib23220/Documents/Descriptors/centered_files/high_ap/FF/" + [f for f in os.listdir("/users/mib23220/Documents/Descriptors/centered_files/high_ap/FF/") if f.endswith('.xtc')][0]
	u = mda.Universe(gro, xtc)
	frame_idx = 1
	u.trajectory[frame_idx]
	peptides = u.select_atoms("name BB")
	result = calculate_adi(peptides, u.dimensions[:3], dynamic_cutoff=True)
	# Basic assertions
	assert 'clusters' in result
	assert isinstance(result['clusters'], list)
	assert result['r_cut'] > 0
	print(f"Frame 5000: Found {len(result['clusters'])} clusters, r_cut={result['r_cut']}")
