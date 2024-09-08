#!/bin/bash

# Hardcoded input file name
input_file="input.txt"

# Temporary file to store the result
temp_file="temp_output.txt"

# Check if the input file exists
if [ ! -f "$input_file" ]; 
then
    echo "Error: Input file '$input_file' not found."
    exit 1
fi

# Remove duplicate lines and save to a temporary file
sort "$input_file" | uniq > "$temp_file"

# Check if the sort and uniq commands were successful
if [ $? -ne 0 ]; 
then
    echo "Error: Failed to process the file."
    rm -f "$temp_file"
    exit 2
fi

# Replace the original file with the deduplicated content
mv "$temp_file" "$input_file"

echo "Duplicate lines have been removed from '$input_file'."
