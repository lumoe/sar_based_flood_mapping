#!/bin/bash

# Set input directory
input_dir="data/"

# Loop through each .tif file in the input directory and its subdirectories
find "$input_dir" -name "*.tif" -print0 | while read -d $'\0' file; do
    # Get the directory, filename and extension of the input file
    dir=$(dirname "$file")
    filename=$(basename "$file")
    extension="${filename##*.}"

    # Create the output filename and path
    output_filename="${filename%.*}_TILED.tif"
    output_file="$dir/$output_filename"

    # Run gdal_translate with the desired settings
    gdal_translate -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 "$file" "$output_file"
done
