#!/usr/bin/env bash


while getopts ":hi:g:d:" opt; do
    case $opt in
        h)
            echo "-i image_name -g gpu_ids -d description"
            exit 0
            ;;
        i)
#            echo "-i triggered"
            image=$OPTARG
            ;;
        g)
#            echo "-g triggered"
            gpu_ids=$OPTARG
            ;;
        d)
#            echo "-d triggered"
            echo $OPTARG
            descrip=$OPTARG
            ;;
        *)
            echo "Invalid option -$OPTARG"
            exit 1
            ;;
  esac
done

if [[ -z "$image" ]] || [[ -z "$gpu_ids" ]] || [[ -z "$descrip" ]]
then
      echo "insufficient input"
      exit 1
fi

echo "Docker image: $image"
echo "Selected GPUs: $gpu_ids"
echo "Description: $descrip"

IFS=',' read -ra gpu_ids_expanded <<< "$gpu_ids"

j=$(expr ${#gpu_ids_expanded[@]} - 1)
gpu_ids_seen_by_container=$(seq -s " " 0 ${j})

uid=$(id -u)
gid=(id -g)

set -x
docker run -u ${uid}:${gid} \
           -v /home/cs4li/Dev/deep_ekf_vio:/scratch/cs4li/deep_ekf_vio \
           -v /home/cs4li/Dev/KITTI:/scratch/cs4li/KITTI \
           -v /home/cs4li/Dev/EUROC:/scratch/cs4li/EUROC \
           -e NVIDIA_VISIBLE_DEVICES=${gpu} \
           --shm-size 128g --runtime=nvidia --rm ${image} \
           python3 /home/cs4li/Dev/deep_ekf_vio/main.py \
           --description ${descrip} --gpu_id ${gpu_ids_seen_by_container}
set +x