
import numpy as np
from MDAnalysis.analysis import rdf

def compute_rdf(peptides, rdf_range=(0.0, 30.0), nbins=100):
	rdf_analysis = rdf.InterRDF(peptides, peptides, nbins=nbins, range=rdf_range, exclusion_block=(1, 1))
	rdf_analysis.run()
	return rdf_analysis.bins, rdf_analysis.rdf

def find_first_minimum(r, g_r):
	peak_idx = np.argmax(g_r)
	if peak_idx+1 >= len(g_r):
		return r[peak_idx]
	min_idx = peak_idx + np.argmin(g_r[peak_idx+1:]) + 1
	return r[min_idx] if min_idx < len(r) else r[peak_idx]
