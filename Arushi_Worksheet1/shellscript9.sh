#!/bin/bash

# Get the current hour in 24-hour format
current_hour=$(date +%H)

# Convert the string to an integer
current_hour=$((10#$current_hour))

# Determine the greeting based on the current hour
if [ $current_hour -ge 5 ] && [ $current_hour -lt 12 ]; 
then
    greeting="Good morning!"
elif [ $current_hour -ge 12 ] && [ $current_hour -lt 18 ]; 
then
    greeting="Good afternoon!"
else
    greeting="Good night!"
fi

# Print the greeting
echo $greeting
