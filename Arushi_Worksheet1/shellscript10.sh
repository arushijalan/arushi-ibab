#!/bin/bash

# Hardcoded DNA sequence for testing
DNA_SEQUENCE="ACGTACGTACGTACGT"

# Function to count nucleotides
count_nucleotides() {
    local sequence=$1
    local length=${#sequence}
    
    # Initialize counters
    local count_A=0
    local count_C=0
    local count_G=0
    local count_T=0
    
    # Count nucleotides
    for (( i=0; i<$length; i++ )); 
    do
        case ${sequence:$i:1} in
            A) ((count_A++)) ;;
            C) ((count_C++)) ;;
            G) ((count_G++)) ;;
            T) ((count_T++)) ;;
            *)
                echo "Error: Invalid character '${sequence:$i:1}' found in DNA sequence."
                return 1
                ;;
        esac
    done
    
    # Print results
    echo "A: $count_A"
    echo "C: $count_C"
    echo "G: $count_G"
    echo "T: $count_T"
}

# Call the function with the hardcoded DNA sequence
echo "Analyzing DNA sequence: $DNA_SEQUENCE"
if ! count_nucleotides "$DNA_SEQUENCE"; 
then
    echo "The DNA sequence contains invalid characters."
    exit 1
fi

# Test with an invalid sequence
INVALID_SEQUENCE="ACGTACGTACGTACGTX"
echo -e "\nTesting with invalid sequence: $INVALID_SEQUENCE"
if ! count_nucleotides "$INVALID_SEQUENCE"; 
then
    echo "The DNA sequence contains invalid characters."
    exit 1
fi

