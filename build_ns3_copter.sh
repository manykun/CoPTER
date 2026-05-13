#!/bin/sh

# Step 0: Change directory to ns-3.33
cd ns-3.33 || { echo "Failed to enter ns-3.33 directory."; exit 1; }

# Step 1: Backup the protobuf directory
echo "Step 1. Backing up protobuf include directory, please provide your password if prompted..."
sudo mv /usr/local/include/google/protobuf/ /usr/local/include/google/protobuf_bak/

# Step 2: Configure with waf
echo "Step 2. Configuring with waf and building protobuf for ns3-gym..."
./waf configure --enable-examples --enable-mpi > ../build_ns3_copter.log 2>&1
if [ $? -ne 0 ]; then
    echo "Error: ./waf configure failed. See build_ns3_copter.log for details."
    exit 1
fi

# Step 3: Build with waf
echo "Step 3. Building ns-3 with waf..."
./waf build >> ../build_ns3_copter.log 2>&1
if [ $? -ne 0 ]; then
    echo "Error: ./waf build failed. See build_ns3_copter.log for details."
    echo "\033[43;38m Please make sure the protobuf compiler (protoc and include) used is version 3.6.1, and the Python protobuf package is version 3.20.*. \033[0m"
    exit 1
fi

# Step 4: Install ns3gym module
echo "Step 4. Installing ns3gym Python package via pip..."
pip3 install --user ./contrib/opengym/model/ns3gym >> ../build_ns3_copter.log 2>&1
if [ $? -ne 0 ]; then
    echo "Error: pip install failed."
    exit 1
fi

# Step 5: Restore protobuf directory
echo "Step 5. Restoring protobuf include directory..."
sudo mv /usr/local/include/google/protobuf_bak/ /usr/local/include/google/protobuf/

# echo in green color
echo "\033[32mCoPTER Safe Build: all steps completed successfully! See build_ns3_copter.log for details.\033[0m"