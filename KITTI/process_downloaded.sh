#!/bin/bash

files=(
00:2011_10_03_drive_0027
01:2011_10_03_drive_0042
02:2011_10_03_drive_0034
03:2011_09_26_drive_0067
04:2011_09_30_drive_0016
05:2011_09_30_drive_0018
06:2011_09_30_drive_0020
07:2011_09_30_drive_0027
08:2011_09_30_drive_0028
09:2011_09_30_drive_0033
10:2011_09_30_drive_0034
)
#echo 'start downloading trained model......'
#wget 'https://www.polybox.ethz.ch/index.php/s/90OlHg6KWBzG6gR'
#echo 'model downloading finished! start downloading raw images....'

mkdir 'images'

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                rename=${i:0:2}
                shortname=${i:3}'_sync.zip'
                fullname=${i:3}'/'${i:3}'_sync.zip'
        else
                $i=${i:(-3)}
                shortname=$i
                fullname=$i
        fi

        # Remove image00 image01 image02, rename dir
        if [ ${i:(-3)} != "zip" ]
        then
                dirn=${i:3}'_sync''/'
                #mv $dirn 'images/'$rename
                mv $dirn'image_02/data' 'images/'$rename
        fi
echo 'all downloading done!'
done

