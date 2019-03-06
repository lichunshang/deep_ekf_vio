#!/usr/bin/env bash

trap "exit" SIGINT SIGTERM

SCRIPT=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPT")
MATLAB_SCRIPTS_DIR=$SCRIPTDIR/gps2local
KITTI_DATASET_DIR='/home/cs4li/Dev/KITTI/dataset'
OUTPUT_DIR="$SCRIPTDIR/../data"

echo "SCRIPT DIR: $SCRIPTDIR"
echo "KITTI Dataset Dir: $KITTI_DATASET_DIR"
echo "OUTPUT DIR: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# use matlab to convert
gps2pose_convert=(
2011_10_03/2011_10_03_drive_0027_extract
2011_10_03/2011_10_03_drive_0042_extract
2011_10_03/2011_10_03_drive_0034_extract
2011_09_30/2011_09_30_drive_0016_extract
2011_09_30/2011_09_30_drive_0018_extract
2011_09_30/2011_09_30_drive_0020_extract
2011_09_30/2011_09_30_drive_0027_extract
2011_09_30/2011_09_30_drive_0028_extract
2011_09_30/2011_09_30_drive_0033_extract
2011_09_30/2011_09_30_drive_0034_extract
)

if [[ -z "$1" ]]
then
for SEQ_DIR in "${gps2pose_convert[@]}"
do
    echo "============================================="
    echo "Convert GPS lat/long/alt yaw/pitch/roll to Cartesian using MATLAB Script..."
	matlab -nodisplay -nodesktop -sd $MATLAB_SCRIPTS_DIR -r "run_OxtsToPoseTxt('$KITTI_DATASET_DIR/$SEQ_DIR'); exit"
done
fi

# now generate the sequences
seq_data=(
2011_10_03/2011_10_03_drive_0027_extract K00_0 20 351
2011_10_03/2011_10_03_drive_0027_extract K00_1 368 1917
2011_10_03/2011_10_03_drive_0027_extract K00_2 1933 1959
2011_10_03/2011_10_03_drive_0027_extract K00_3 1975 2119
2011_10_03/2011_10_03_drive_0027_extract K00_4 2136 2279
2011_10_03/2011_10_03_drive_0027_extract K00_5 2296 2691
2011_10_03/2011_10_03_drive_0027_extract K00_6 2707 2948
2011_10_03/2011_10_03_drive_0027_extract K00_7 2986 4548
#
2011_10_03/2011_10_03_drive_0042_extract K01 5 1105
#
2011_10_03/2011_10_03_drive_0034_extract K02_0 5 3160
2011_10_03/2011_10_03_drive_0034_extract K02_1 3176 3267
2011_10_03/2011_10_03_drive_0034_extract K02_2 3284 3312
2011_10_03/2011_10_03_drive_0034_extract K02_3 3329 4667
#
2011_09_30/2011_09_30_drive_0016_extract K04 5 283
#
2011_09_30/2011_09_30_drive_0018_extract K05_0 5 1503
2011_09_30/2011_09_30_drive_0018_extract K05_1 1520 2766
#
2011_09_30/2011_09_30_drive_0020_extract K06 18 1108
2011_09_30/2011_09_30_drive_0027_extract K07 5 1110
2011_09_30/2011_09_30_drive_0028_extract K08 1400 5181
2011_09_30/2011_09_30_drive_0033_extract K09 5 1598
2011_09_30/2011_09_30_drive_0034_extract K10 5 1229
)

NUM_SEQS=`expr ${#seq_data[@]} / 4 - 1`

for i in `seq 0 $NUM_SEQS`
do
    SEQ_DIR=${seq_data[`expr "$i" \* 4`]}
    SEQ_NAME=${seq_data[`expr "$i" \* 4 + 1`]}
    START=${seq_data[`expr "$i" \* 4 + 2`]}
    END=${seq_data[`expr "$i" \* 4 + 3`]}
    echo "Preprocessing converted seq_data using python script..."
    python3 $SCRIPTDIR/../exec.py preprocess_kitti_raw $KITTI_DATASET_DIR/$SEQ_DIR $OUTPUT_DIR/$SEQ_NAME $START $END
    echo
done