#!/bin/bash

# Set the threshold percentage
THRESHOLD=70

# Get the disk usage percentage of the partition where $HOME is located
disk_usage=$(df -h "$HOME" | awk 'NR==2 {print $5}' | sed 's/%//')

# Check if the usage exceeds the threshold
if [ "$disk_usage" -gt "$THRESHOLD" ]; 
then
  echo "Alert: Disk usage for the $HOME directory has exceeded ${THRESHOLD}% (Current usage: ${disk_usage}%)"
else
  echo "Disk usage for the $HOME directory is within limits (Current usage: ${disk_usage}%)"
fi
