#!/bin/bash

# Function to check if a file is executable
is_executable() {
    if [ -x "$1" ] && [ -f "$1" ]; 
    then
        return 0
    else
        return 1
    fi
}

# Function to search for executables in a directory
search_executables() {
    local dir="$1"
    if [ -d "$dir" ]; 
    then
        for file in "$dir"/*; 
        do
            if is_executable "$file"; 
            then
                echo "$file"
            fi
        done
    fi
}

# Main script

echo "Searching for executable files in the system..."
echo "This may take a while and might require sudo privileges for some directories."

# List of directories to search
directories=(
    "/bin"
    "/usr/bin"
    "/usr/local/bin"
    "/sbin"
    "/usr/sbin"
    "/usr/local/sbin"
    "$HOME/bin"
    "$HOME/.local/bin"
)

# Search each directory
for dir in "${directories[@]}"; 
do
    echo "Searching in $dir..."
    search_executables "$dir"
done

echo "Search completed."

echo "To search for executables in other directories, you can modify the 'directories' array in this script."

