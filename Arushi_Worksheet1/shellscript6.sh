#!/bin/bash

# Hardcoded directory path (replace with the path you want to search)
search_dir="/path/to/your/directory"

# Output file to store empty subfolder names
output_file="empty_subfolders.txt"

# Check if the search directory exists
if [ ! -d "$search_dir" ]; 
then
    echo "Error: Directory '$search_dir' not found."
    exit 1
fi

# Find empty subfolders and write their names to the output file
find "$search_dir" -type d -empty | while read -r dir; 
do
    echo "${dir#$search_dir/}" >> "$output_file"
done

# Check if any empty subfolders were found
if [ -s "$output_file" ]; 
then
    echo "Empty subfolders have been written to '$output_file'."
else
    echo "No empty subfolders found in '$search_dir'."
    rm -f "$output_file"  # Remove the empty output file
fi
