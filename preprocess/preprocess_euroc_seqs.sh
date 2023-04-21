#!/usr/bin/env bash

trap "exit" SIGINT SIGTERM

SCRIPT=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPT")
EUROC_DATASET_DIR='/mnt/data/teamAI/quyen/deep_vio/deep_ekf_vio/data'
OUTPUT_DIR="$SCRIPTDIR/../data"

echo "SCRIPT DIR: $SCRIPTDIR"
echo "EUROC Dataset Dir: $EUROC_DATASET_DIR"
echo "OUTPUT DIR: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# now generate the sequences
seq_data=(
MH_01_easy       MH_01   421 621
MH_02_easy       MH_02   534 594
MH_03_medium     MH_03   0 40
MH_04_difficult  MH_04   0 20
MH_05_difficult  MH_05   0 60
V1_01_easy       V1_01   0 90
V1_02_medium     V1_02   0 70
V1_03_difficult  V1_03   0 90
V2_01_easy       V2_01   0 60
V2_02_medium     V2_02   0 40
V2_03_difficult  V2_03   0 70
)

NUM_SEQS=`expr ${#seq_data[@]} / 4 - 1`

for i in `seq 0 $NUM_SEQS`
do
    SEQ_DIR=${seq_data[`expr "$i" \* 4`]}
    SEQ_NAME=${seq_data[`expr "$i" \* 4 + 1`]}
    CAM_STILL_START=${seq_data[`expr "$i" \* 4 + 2`]}
    CAM_STILL_END=${seq_data[`expr "$i" \* 4 + 3`]}

#    echo $SEQ_DIR
#    echo $SEQ_NAME
#    echo $CAM_STILL_START
#    echo $CAM_STILL_END

    echo "Preprocessing converted seq_data using python script..."
    python3 $SCRIPTDIR/../exec.py preprocess_euroc $EUROC_DATASET_DIR/$SEQ_DIR/mav0 $OUTPUT_DIR/$SEQ_NAME $CAM_STILL_START $CAM_STILL_END
    echo
done