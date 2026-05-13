#!/bin/bash

NS3_DIR=../ns-3.33
NS3_BLD_DIR=$NS3_DIR/build
NS3_LIB_DIR=$(realpath $NS3_BLD_DIR)/lib
LOG_LEV=level_all

# Create output directory if it doesn't exist
mkdir -p output

# List of configuration files to process
CONFIG_FILES=(
    "acc_mixed_1_long_DCQCN_SECN"
    "acc_mixed_1_long_HPCC_SECN"
)

# Check if any config file is missing
for CONFIG in "${CONFIG_FILES[@]}"; do
    if [ ! -f "./mix/${CONFIG}.conf" ]; then
        echo "Error: Configuration file './mix/${CONFIG}.conf' not found"
        exit 1
    fi
done

# Set environment variables
export LD_LIBRARY_PATH=$NS3_LIB_DIR:$LD_LIBRARY_PATH
export NS_LOG=CongestionControlSimulator=$LOG_LEV

# Run simulations concurrently
for CONFIG in "${CONFIG_FILES[@]}"; do
    OUTPUT_FILE="output/${CONFIG}.output"
    echo "Starting simulation with $CONFIG..."
    (
        $NS3_BLD_DIR/scratch/copter-sim "./mix/${CONFIG}.conf" > "$OUTPUT_FILE" 2>&1
        echo "Simulation completed for $CONFIG, output saved to $OUTPUT_FILE"
    ) &  # Run in background
done

# Wait for all background processes to complete
wait

echo "All simulations completed"