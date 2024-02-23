#!/bin/bash

# Set the control mesh directories and test options
control_mesh_dirs=("/home/yd21/Documents/Nasreddin/template/control_mesh-lv.obj" "/home/yd21/Documents/Nasreddin/template/control_mesh-myo.obj" "/home/yd21/Documents/Nasreddin/template/control_mesh-rv.obj")
test_options=("sct" "cap")

# Loop through the control mesh directories
for dir in "${control_mesh_dirs[@]}"; do
    # Loop through the test options
    for on in "${test_options[@]}"; do
        # Run the Python script with the current control mesh directory and test option
        python main_stationary.py --control_mesh_dir "$dir" --test_on "$on"
    done
done
