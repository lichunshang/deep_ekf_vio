#!/usr/bin/env bash

trap "exit" SIGINT SIGTERM

SCRIPT=$(readlink -f "$0")
SCRIPTDIR=$(dirname "$SCRIPT")
TUM_DATASET_DIR='/mnt/data/teamAI/duy/data/TUM'
OUTPUT_DIR="$SCRIPTDIR/../data"

echo "SCRIPT DIR: $SCRIPTDIR"
echo "TUM Dataset Dir: $TUM_DATASET_DIR"
echo "OUTPUT DIR: $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

# now generate the sequences
seq_data=(
dataset-corridor1_512_16    C1  0   40
dataset-corridor2_512_16    C2  0   40
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
    python3 $SCRIPTDIR/../exec.py preprocess_tum $TUM_DATASET_DIR/$SEQ_DIR/mav0 $OUTPUT_DIR/$SEQ_NAME $CAM_STILL_START $CAM_STILL_END
    echo
done