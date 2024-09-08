#!/bin/bash

# CSV file name
csv_file="users.csv"

# Check if the file exists
if [ ! -f "$csv_file" ]; 
then
    echo "Error: $csv_file not found."
    exit 1
fi

# Read the CSV file line by line
while IFS=',' read -r userid rest; 
do
    # Split the rest of the line by colon
    IFS=':' read -r username userdept <<< "$rest"
    
    # Display the information
    echo "UserID: $userid"
    echo "Username: $username"
    echo "Department: $userdept"
    echo "-------------------"
done < "$csv_file"
