# Use MDAnalysis AtomGroup.wrap for PBC handling
def wrap_atoms(atomgroup, compound='segments'):
    """
    Wrap atoms into the primary box using MDAnalysis AtomGroup.wrap.
    Args:
        atomgroup: MDAnalysis AtomGroup
        compound: 'segments', 'residues', or 'atoms' (default: 'segments')
    """
    atomgroup.wrap(compound=compound)
