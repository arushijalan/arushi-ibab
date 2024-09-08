#!/bin/bash

# Set the threshold value
threshold=90

# Loop through numbers 1 to 100
for i in {1..100}
do
    # Print the current number
    echo $i
    
    # Check if the number is greater than the threshold
    if [ $i -gt $threshold ]
    then
        echo "The number $i is greater than the threshold ($threshold)"
    fi
done
