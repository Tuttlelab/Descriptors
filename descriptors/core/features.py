
def build_feature_row(frame, cluster, adi, sfi, vfi, ffi, tfi):
	row = {
		'frame': frame,
		'cluster': cluster,
		# ADI
		'adi_n_clusters': adi.get('n_clusters'),
		'adi_r_cut': adi.get('r_cut'),
		'adi_size': adi.get('cluster_sizes', [None])[cluster],
		'adi_cluster_sizes': adi.get('cluster_sizes'),
		'adi_clusters': adi.get('clusters'),
		# SFI
		'sfi_n_sheets': sfi.get('n_sheets'),
		'sfi_sheet_sizes': sfi.get('sheet_sizes'),
		'sfi_clusters': sfi.get('clusters'),
		# VFI
		'vfi_is_vesicle': vfi.get('is_vesicle'),
		'vfi_sphericity': vfi.get('sphericity'),
		'vfi_hollow': vfi.get('hollow'),
		'vfi_density_profile': vfi.get('density_profile'),
		'vfi_bin_edges': vfi.get('bin_edges'),
		# FFI
		'ffi_is_fiber': ffi.get('is_fiber'),
		'ffi_shape_ratios': ffi.get('shape_ratios'),
		'ffi_eigvals': ffi.get('eigvals'),
		# TFI
		'tfi_is_tube': tfi.get('is_tube'),
		'tfi_radial_std': tfi.get('radial_std'),
	}
	return row
