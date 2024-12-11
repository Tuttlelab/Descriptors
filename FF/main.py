#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from datetime import datetime
from collections import defaultdict
import logging
import numpy as np

# Import analysis modules
from center import load_and_process_trajectory as center_trajectory
from cluster import analyze_trajectory as cluster_trajectory
from sfi import analyze_clusters as analyze_sfi
from vfi import analyze_clusters as analyze_vfi
from ffi import analyze_clusters as analyze_ffi
from tfi import analyze_clusters as analyze_tfi
from plot import analyze_and_plot_evolution as analyze_and_plot

# Add these imports at the top with other imports
from sfi import (
    PLANARITY_THRESHOLD, CURVATURE_THRESHOLD,
    MIN_SHEET_SIZE, MIN_ATOMS_SPHERICITY
)
from vfi import (
    SPHERICITY_THRESHOLD, HOLLOWNESS_THRESHOLD,
    ASPHERICITY_THRESHOLD, ACYLINDRICITY_THRESHOLD
)
from ffi import (
    SHAPE_RATIO_THRESHOLD, ALIGNMENT_STD_THRESHOLD,
    FOP_THRESHOLD_POSITIVE, MIN_LENGTH_RATIO,
    RADIUS_VARIATION_THRESHOLD, MIN_CYLINDRICAL_SCORE
)
from tfi import (
    RADIAL_THRESHOLD, ANGULAR_UNIFORMITY_THRESHOLD,
    ASPHERICITY_THRESHOLD as TUBE_ASPHERICITY_THRESHOLD,
    RATIO_THRESHOLD, TUBE_SEGMENT_RATIO_THRESHOLD
)

def setup_logging(output_dir):
    """Setup logging with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'analysis_{timestamp}.log')

    # Configure console handler to only show INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Configure file handler for all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Set MDAnalysis logger to INFO level to suppress DEBUG messages
    logging.getLogger('MDAnalysis').setLevel(logging.INFO)

    # Get loggers for other modules and set their levels
    logging.getLogger('sfi').setLevel(logging.DEBUG)
    logging.getLogger('ffi').setLevel(logging.DEBUG)

    return logging.getLogger('main')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Peptide Assembly Analysis Pipeline')

    # Main operation flags
    parser.add_argument('--center', action='store_true', help='Run centering step')
    parser.add_argument('--cluster', action='store_true', help='Run clustering step')
    parser.add_argument('--analyze', action='store_true', help='Run shape analysis')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--all', action='store_true', help='Run full pipeline')

    # Input/Output
    parser.add_argument('-t', '--topology', required=True, help='Input topology file (.gro)')
    parser.add_argument('-x', '--trajectory', required=False, help='Input trajectory file (.xtc)')
    parser.add_argument('-o', '--output', default='results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    # Frame selection
    parser.add_argument('--first', type=int, default=0, help='First frame to analyze (default: 0)')
    parser.add_argument('--last', type=int, default=None, help='Last frame to analyze (default: all frames)')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride interval (default: 1)')

    # Clustering parameters
    parser.add_argument('--cluster_cutoff', type=float, default=4.5,
                       help='Cutoff distance for clustering in nm (default: 4.5)')
    parser.add_argument('--min_peptides', type=int, default=20,
                       help='Minimum number of peptides for a valid cluster (default: 20)')

    args = parser.parse_args()

    # Validate frame arguments
    if args.first < 0:
        raise ValueError("First frame must be non-negative")
    if args.last is not None and args.last < args.first:
        raise ValueError("Last frame must be greater than first frame")
    if args.stride < 1:
        raise ValueError("Stride must be positive")

    return args

def clean_clusters_directory(output_dir):
    """Clean up old cluster files before new analysis"""
    clusters_dir = os.path.join(output_dir, "clusters")
    if os.path.exists(clusters_dir):
        for file in os.listdir(clusters_dir):
            if file.endswith('.gro'):
                os.remove(os.path.join(clusters_dir, file))

def ensure_analysis_directories(output_dir):
    """Create necessary directories for analysis results"""
    subdirs = ['clusters', 'logs', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

def classify_frame_aggregates(cluster_files, args):
    """Analyze clusters with mutual exclusion"""
    if not cluster_files:
        return "no_agg"

    shape_counts = defaultdict(int)
    total_clusters = len(cluster_files)

    for cluster_file in cluster_files:
        # Try each shape analysis
        is_determined = False

        # Sheet analysis
        sheet_results = analyze_sfi([cluster_file], args.min_peptides)
        if any(r.get('is_sheet') for r in sheet_results):
            shape_counts['sheet'] += 1
            is_determined = True

        # Fiber analysis
        fiber_results = analyze_ffi([cluster_file], args.min_peptides)
        if any(r.get('is_fiber') for r in fiber_results):
            shape_counts['fiber'] += 1
            is_determined = True

        # Tube analysis
        tube_results = analyze_tfi([cluster_file], args.min_peptides)
        if any(r.get('is_tube') for r in tube_results):
            shape_counts['tube'] += 1
            is_determined = True

        # Vesicle analysis
        vesicle_results = analyze_vfi([cluster_file], args.min_peptides)
        if any(r.get('is_vesicle') for r in vesicle_results):
            shape_counts['vesicle'] += 1
            is_determined = True

        if not is_determined:
            shape_counts['undetermined'] += 1

    # Determine dominant shape
    if not shape_counts:
        return "no_agg"
    if shape_counts['undetermined'] == total_clusters:
        return "undetermined"

    # Remove undetermined from consideration for dominant shape
    del shape_counts['undetermined']
    if shape_counts:
        dominant_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
        return dominant_shape

    return "undetermined"

def find_cluster_files(output_dir, first_frame=None, last_frame=None):
    """Find all cluster files in frame-specific directories"""
    cluster_files = []
    clusters_dir = os.path.join(output_dir, "clusters")

    if os.path.exists(clusters_dir):
        # Look for frame directories (f*)
        for frame_dir in os.listdir(clusters_dir):
            if frame_dir.startswith('f'):
                try:
                    frame_num = int(frame_dir[1:])  # Extract number after 'f'
                    if (first_frame is None or frame_num >= first_frame) and \
                       (last_frame is None or frame_num <= last_frame):
                        frame_path = os.path.join(clusters_dir, frame_dir)
                        if os.path.isdir(frame_path):
                            for cluster_file in os.listdir(frame_path):
                                if cluster_file.endswith('.gro'):
                                    cluster_files.append(os.path.join(frame_path, cluster_file))
                except ValueError:
                    continue  # Skip directories that don't match fXXXX pattern

    return sorted(cluster_files)

def save_shape_results(results, output_dir, frame_num, logger):
    """Save shape analysis results to a file"""
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Create a results file for this frame
    output_file = os.path.join(results_dir, f'frame{frame_num}_clusters.txt')

    with open(output_file, 'w') as f:
        for result in results:
            cluster_num = result.get('cluster_num', 'unknown')
            cluster_size = result.get('size', 0)
            shapes = result.get('shapes', [])  # Get the true shapes list

            shapes_text = ', '.join(shape.capitalize() for shape in shapes) if shapes else 'Undetermined'
            msg = f"Cluster {cluster_num} with {cluster_size} beads, identified as: {shapes_text}"
            logger.info(msg)
            f.write(f"{msg}\n")

def analyze_frame_clusters(frame_cluster_files, min_peptides, logger, args, frame_num):
    """Analyze all clusters in a single frame and determine dominant shape"""
    frame_results = {
        'sheets': 0, 'fibers': 0, 'vesicles': 0, 'tubes': 0,
        'total_peptides': 0, 'shape_counts': {},
        'largest_cluster_shape': 'undetermined',
        'cluster_results': []  # Store individual cluster results
    }

    # Track largest cluster
    largest_cluster_size = 0
    largest_cluster_shape = 'undetermined'

    for cluster_file in frame_cluster_files:
        try:
            cluster_num = os.path.basename(cluster_file).split('_')[0].replace('cluster', '')

            shapes = []
            all_metrics = {}

            # Sheet analysis
            sheet_results = analyze_sfi([cluster_file], min_peptides)
            if sheet_results and sheet_results[0].get('metrics'):
                metrics = sheet_results[0]['metrics']
                if sheet_results[0].get('is_sheet', False):
                    shapes.append('sheet')
                all_metrics['sheet'] = metrics

            # Fiber analysis
            fiber_results = analyze_ffi([cluster_file], min_peptides)
            if fiber_results and fiber_results[0].get('metrics'):
                metrics = fiber_results[0]['metrics']
                if fiber_results[0].get('is_fiber', False):
                    shapes.append('fiber')
                all_metrics['fiber'] = metrics

            # Vesicle analysis
            vesicle_results = analyze_vfi([cluster_file], min_peptides)
            if vesicle_results and vesicle_results[0].get('metrics'):
                metrics = vesicle_results[0]['metrics']
                if vesicle_results[0].get('is_vesicle', False):
                    shapes.append('vesicle')
                all_metrics['vesicle'] = metrics

            # Tube analysis
            tube_results = analyze_tfi([cluster_file], min_peptides)
            if tube_results and tube_results[0].get('metrics'):
                metrics = tube_results[0]['metrics']
                if tube_results[0].get('is_tube', False):
                    shapes.append('tube')
                all_metrics['tube'] = metrics

            # Get cluster size from any valid result
            cluster_size = next((r['size'] for results in [sheet_results, fiber_results, vesicle_results, tube_results]
                               for r in results if 'size' in r), 0)

            # Round metrics before logging
            for shape_type, metrics in all_metrics.items():
                if isinstance(metrics, dict):
                    all_metrics[shape_type] = {k: round(float(v), 1) if isinstance(v, (float, np.floating)) else v
                                             for k, v in metrics.items()}

            # Create cluster result with all shape flags
            cluster_result = {
                'cluster_num': cluster_num,
                'size': cluster_size,
                'shapes': shapes,
                'metrics': all_metrics
            }

            # Log results
            logger.info(f"\nCluster {cluster_num} analysis:")
            for shape_type, metrics in all_metrics.items():
                logger.info(f"{shape_type.capitalize()} metrics: {metrics}")

            # Only show shapes that were determined to be true
            true_shapes = []
            if sheet_results and sheet_results[0].get('is_sheet', False):
                true_shapes.append('sheet')
            if fiber_results and fiber_results[0].get('is_fiber', False):
                true_shapes.append('fiber')
            if vesicle_results and vesicle_results[0].get('is_vesicle', False):
                true_shapes.append('vesicle')
            if tube_results and tube_results[0].get('is_tube', False):
                true_shapes.append('tube')

            current_shape = 'undetermined' if not true_shapes else ', '.join(true_shapes)
            logger.info(f"Classification: {current_shape}")

            # Create cluster result with only true shapes
            cluster_result = {
                'cluster_num': cluster_num,
                'size': cluster_size,
                'shapes': true_shapes,  # Use true_shapes instead of shapes
                'metrics': all_metrics
            }

            # Update frame results based on true shapes
            for shape in true_shapes:  # Use true_shapes instead of shapes
                frame_results[f'{shape}s'] += cluster_size
                frame_results['shape_counts'][shape] = frame_results['shape_counts'].get(shape, 0) + 1

            # Update largest cluster information
            if cluster_size > largest_cluster_size:
                largest_cluster_size = cluster_size
                largest_cluster_shape = current_shape

            frame_results['cluster_results'].append(cluster_result)

        except Exception as e:
            logger.error(f"Error analyzing cluster {cluster_file}: {str(e)}", exc_info=True)
            continue

    frame_results['largest_cluster_shape'] = largest_cluster_shape
    frame_results['total_peptides'] = (frame_results['sheets'] + frame_results['fibers'] +
                                     frame_results['vesicles'] + frame_results['tubes'])

    # Save detailed results for this frame
    save_shape_results(frame_results['cluster_results'], args.output, frame_num, logger)

    return frame_results

def save_analysis_data(frame_results, output_dir):
    """Save analysis data for plotting"""
    results_dir = os.path.join(output_dir, "analysis")
    os.makedirs(results_dir, exist_ok=True)

    # Convert all frame numbers to integers
    frame_results = {int(frame): results for frame, results in frame_results.items()}

    # Save frame-by-frame data
    with open(os.path.join(results_dir, 'shape_evolution.csv'), 'w') as f:
        f.write('Frame,Sheets,Fibers,Vesicles,Tubes,Total,LargestClusterShape\n')
        for frame, results in sorted(frame_results.items()):
            # Get counts from shape_counts dictionary, default to 0 if not present
            sheet_count = results['shape_counts'].get('sheet', 0)
            fiber_count = results['shape_counts'].get('fiber', 0)
            vesicle_count = results['shape_counts'].get('vesicle', 0)
            tube_count = results['shape_counts'].get('tube', 0)

            f.write(f"{frame},{sheet_count},{fiber_count},"
                   f"{vesicle_count},{tube_count},{results['total_peptides']},"
                   f"{results['largest_cluster_shape']}\n")

def main():
    args = parse_arguments()
    ensure_analysis_directories(args.output)
    logger = setup_logging(args.output)

    # Set flags if --all is specified
    if args.all:
        args.center = args.cluster = args.analyze = args.plot = True

    # Track files for pipeline
    if args.trajectory:
        centered_files = (args.topology, args.trajectory)
    else:
        centered_files = (args.topology, None)

    # Clean up old cluster files if doing clustering
    if args.cluster:
        clean_clusters_directory(args.output)

    # Check for existing cluster files using new naming pattern
    clusters_dir = os.path.join(args.output, "clusters")
    if os.path.exists(clusters_dir):
        cluster_files = sorted([
            os.path.join(clusters_dir, f) for f in os.listdir(clusters_dir)
            if f.endswith('.gro') and f.startswith('frame')
        ])
    else:
        cluster_files = []

    shape_results = defaultdict(lambda: defaultdict(list))

    try:
        # Centering step
        if args.center:
            if centered_files[1]:
                centered_files = center_trajectory(
                    args.topology,
                    args.trajectory,
                    args.output,
                    "FF",  # or get from filename
                    args.first,
                    args.last,
                    args.stride
                )
                logger.info("Centering completed successfully")
            else:
                logger.info("No trajectory file provided. Skipping centering step")

        # Clustering step - only if no cluster files exist
        if args.cluster or (args.analyze and not cluster_files):
            if centered_files[1]:
                cluster_files = cluster_trajectory(
                    centered_files[0],
                    centered_files[1],
                    args.first,
                    args.last,
                    args.stride,
                    args.cluster_cutoff,
                    args.output,
                    logger
                )
                logger.info("Clustering completed successfully")
            else:
                logger.info("No trajectory file provided. Skipping clustering step")

        frame_results = {}

        # Analysis step
        if args.analyze:
            cluster_files = find_cluster_files(args.output, args.first, args.last)
            if not cluster_files:
                logger.error("No cluster files found for analysis!")
                sys.exit(1)

            # Group clusters by frame and analyze
            frame_clusters = defaultdict(list)
            for cluster_file in cluster_files:
                frame_dir = os.path.basename(os.path.dirname(cluster_file))
                # Ensure frame number is stored as integer
                frame_num = int(frame_dir[1:])
                frame_clusters[frame_num].append(cluster_file)

            for frame_num, frame_cluster_files in sorted(frame_clusters.items()):
                logger.info(f"Analyzing frame {frame_num}")
                frame_results[frame_num] = analyze_frame_clusters(
                    frame_cluster_files, args.min_peptides, logger, args, frame_num)

            # Save analysis data
            save_analysis_data(frame_results, args.output)

        # Plotting step
        if args.plot:
            if not frame_results:
                logger.error("No analysis results found for plotting!")
                sys.exit(1)
            analyze_and_plot(frame_results, args.output)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()