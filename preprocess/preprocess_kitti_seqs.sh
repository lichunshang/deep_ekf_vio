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

data=(
2011_10_03/2011_10_03_drive_0027_extract K00 368 4548
2011_10_03/2011_10_03_drive_0042_extract K01 5 1105
2011_10_03/2011_10_03_drive_0034_extract K02 5 4667
2011_09_30/2011_09_30_drive_0016_extract K04 5 283
2011_09_30/2011_09_30_drive_0018_extract K05 5 2766
2011_09_30/2011_09_30_drive_0020_extract K06 18 1108
2011_09_30/2011_09_30_drive_0027_extract K07 5 1110
2011_09_30/2011_09_30_drive_0028_extract K08 1400 5181
2011_09_30/2011_09_30_drive_0033_extract K09 5 1598
2011_09_30/2011_09_30_drive_0034_extract K10 5 1229
)

NUM_SEQS=`expr ${#data[@]} / 4 - 1`

for i in `seq 0 $NUM_SEQS`
do
	echo "==================================================================================="
	echo "==================================================================================="
	echo "==================================================================================="
    SEQ_DIR=${data[`expr "$i" \* 4`]}
    SEQ_NAME=${data[`expr "$i" \* 4 + 1`]}
    START=${data[`expr "$i" \* 4 + 2`]}
    END=${data[`expr "$i" \* 4 + 3`]}
    echo "Processing $SEQ_DIR"
    echo "Convert GPS lat/long/alt yaw/pitch/roll to Cartesian using MATLAB Script..."
	matlab -nodisplay -nodesktop -sd $MATLAB_SCRIPTS_DIR -r "run_OxtsToPoseTxt('$KITTI_DATASET_DIR/$SEQ_DIR'); exit"
    echo "Preprocessing converted data using python script..."
    python3 $SCRIPTDIR/../exec.py preprocess_kitti_raw $KITTI_DATASET_DIR/$SEQ_DIR $OUTPUT_DIR/$SEQ_NAME $START $END
    echo
done