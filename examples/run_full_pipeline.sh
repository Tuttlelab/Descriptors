#!/usr/bin/env bash
# ==============================================================================
# Shape Descriptor Toolkit - Example Full Pipeline Execution
# ==============================================================================
# Usage:
#   bash examples/run_full_pipeline.sh <topology.gro> <trajectory.xtc> <out_dir>
# ==============================================================================

set -e

TOPOLOGY="${1:-data/topology.gro}"
TRAJECTORY="${2:-data/trajectory.xtc}"
OUTPUT_DIR="${3:-results/pipeline_output}"

echo "=============================================================================="
echo "Running Shape Descriptor Toolkit Full Analysis Pipeline"
echo "Topology:   ${TOPOLOGY}"
echo "Trajectory: ${TRAJECTORY}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================================================="

# 1. Run Aggregate Dynamics Index (ADI)
python adi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/adi_results"

# 2. Run Sheet Formation Index (SFI)
python sfi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/sfi_results"

# 3. Run Vesicle Formation Index (VFI)
python vfi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/vfi_results"

# 4. Run Tube Formation Index (TFI)
python tfi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/tfi_results"

# 5. Run Fiber Formation Index (FFI)
python ffi_analysis.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/ffi_results"

# 6. Run Multi-Descriptor Shape Tracker
python shape_tracker.py -t "${TOPOLOGY}" -x "${TRAJECTORY}" -o "${OUTPUT_DIR}/tracking_results"

echo "=============================================================================="
echo "Pipeline execution completed successfully!"
echo "Results saved under: ${OUTPUT_DIR}"
echo "=============================================================================="
