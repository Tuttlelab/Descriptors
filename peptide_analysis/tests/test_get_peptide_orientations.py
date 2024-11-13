import os
import pytest
import numpy as np
import MDAnalysis as mda
import sys
print("Python path:", sys.path)
from peptide_analysis.util.geometry import get_peptide_orientations

def test_get_peptide_orientations():
    # Get the directory of the current test file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths to the input files
    gro_file = os.path.join(current_dir, '..', 'input', 'eq_FF1200.gro')
    xtc_file = os.path.join(current_dir, '..', 'input', 'eq_FF1200.xtc')

    # Load the universe with MDAnalysis
    u = mda.Universe(gro_file, xtc_file)

    # Select the aggregate atoms (modify the selection string as needed)
    aggregate_atoms = u.select_atoms('resname 1PHE or resname 2PHE')

    # Ensure that residues are selected
    assert len(aggregate_atoms.residues) > 0, "No residues selected. Check residue name."

    print(f"Number of selected residues: {len(aggregate_atoms.residues)}")
    print("First few residue names:", [res.resname for res in aggregate_atoms.residues[:10]])

    # Select a specific frame for testing (e.g., the first frame)
    try:
        u.trajectory[100]
    except IndexError:
        pytest.fail("Trajectory does not have 101 frames.")

    # Call the function to get orientation vectors
    orientations = get_peptide_orientations(aggregate_atoms, peptide_length=2)

    # Define expected number of orientation vectors
    expected_num_peptides = len(aggregate_atoms.residues) // 2
    assert orientations.shape == (expected_num_peptides, 3), "Orientation array shape mismatch."

    # Example assertion: Check if all orientation vectors are unit vectors or zero vectors
    norms = np.linalg.norm(orientations, axis=1)
    assert np.allclose(norms, 1.0) or np.allclose(norms, 0.0), "Orientation vectors are not normalized correctly."

    # Optional: Compare with precomputed expected orientations
    # expected_orientations = np.load('tests/expected_orientations.npy')
    # np.testing.assert_array_almost_equal(orientations, expected_orientations, decimal=5)
