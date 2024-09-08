#!/bin/bash

# Get the total size of the root directory (in kilobytes)
root_size=$(du -sk / | awk '{print $1}')

# Get the total size of the home directory (in kilobytes)
home_size=$(du -sk /home | awk '{print $1}')

# Calculate the percentage
percentage=$(echo "scale=2; ($home_size/$root_size)*100" | bc)

# Display the result
echo "Home directory usage as a percentage of the root directory: $percentage%"

