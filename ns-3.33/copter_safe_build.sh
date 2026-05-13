#!/bin/sh

# Step 1: Backup the protobuf directory
echo "Backing up protobuf include directory..."
sudo mv /usr/local/include/google/protobuf/ /usr/local/include/google/protobuf_bak/

# Step 2: Configure with waf
echo "Configuring with waf..."
./waf configure --enable-examples --enable-mpi
if [ $? -ne 0 ]; then
    echo "Error: ./waf configure failed."
    exit 1
fi

# Step 3: Build with waf
echo "Building with waf..."
./waf build
if [ $? -ne 0 ]; then
    echo "Error: ./waf build failed."
    echo -e "\033[43;38m Please make sure the protobuf compiler (protoc and include) used is version 3.6.1, and the Python protobuf package is version 3.20.*. \033[0m"
    exit 1
fi

# Step 4: Install ns3gym module
echo "Installing ns3gym via pip..."
pip3 install --user ./contrib/opengym/model/ns3gym
if [ $? -ne 0 ]; then
    echo "Error: pip install failed."
    exit 1
fi

# Step 5: Restore protobuf directory
echo "Restoring protobuf include directory..."
sudo mv /usr/local/include/google/protobuf_bak/ /usr/local/include/google/protobuf/

echo "All steps completed successfully!"