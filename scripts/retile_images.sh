#!/bin/bash

# Set input directory
input_dir="data/"

# Remove all files that exactly contain TILED.tif
find "$input_dir" -name "*_TILED.tif" -type f -delete
find "$input_dir" -name "*_WARPED.tif" -type f -delete

# Loop through each .tif file in the input directory and its subdirectories, the filename may not contain TILED
find "$input_dir" -name "*.tif" -print0 | while read -d $'\0' file; do
    # Get the directory, filename and extension of the input file
    dir=$(dirname "$file")
    filename=$(basename "$file")
    extension="${filename##*.}"

    # Create the output filename and path
    output_filename="${filename%.*}_WARPED.tif"
    output_file="$dir/$output_filename"

    # Run gdal_translate with the desired settings
    # gdal_translate -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 "$file" "$output_file"
    gdalwarp -of VRT "$file" "$output_file" -t_srs "EPSG:4326";
done
