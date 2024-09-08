 #!/bin/bash

# File containing the list of FASTA filenames
input_file="fasta_files.txt"

# URL prefix for downloading FASTA files
url_prefix="http://www.uniprot.org/uniprot/"

# Motifs to search for
motif1="YVDRHPDDTINDYLNSI"
motif2="MGNHTWDHPDIFEILTTK"

# Function to search for motifs in a file
search_motifs() {
    local file=$1
    local motif=$2
    grep -ob "$motif" "$file" | while IFS=: read -r pos match; do
        echo "$file:$pos"
    done
}

# Main loop to process each FASTA file
while read -r filename; 
do
    echo "Processing $filename..."
    
    # Download the FASTA file
    wget -q "${url_prefix}${filename}" -O "$filename"
    
    if [ $? -ne 0 ]; 
    then
        echo "Failed to download $filename"
        continue
    fi
    
    # Search for motifs
    result1=$(search_motifs "$filename" "$motif1")
    result2=$(search_motifs "$filename" "$motif2")
    
    # Output results
    if [ -n "$result1" ]; 
    then
        echo "Motif 1 found in:"
        echo "$result1"
    fi
    
    if [ -n "$result2" ]; 
    then
        echo "Motif 2 found in:"
        echo "$result2"
    fi
    
    if [ -z "$result1" ] && [ -z "$result2" ]; 
    then
        echo "No motifs found in $filename"
    fi
    
    echo "-------------------"
    
    # Clean up: remove the downloaded file
    rm "$filename"
done < "$input_file"
