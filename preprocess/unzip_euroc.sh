#!/usr/bin/env bash
trap "exit" INT

folder=$1
zips_folder=$folder/zips

for zip_file in "$zips_folder"/*
do
    unzip $zip_file -d $folder | pv -l >/dev/null
    rm -rf __MACOSX
    new_dir=$(basename "$zip_file" .zip)
    mv mav0 $new_dir
    echo "Done $new_dir"
done