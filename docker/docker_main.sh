#!/usr/bin/env bash


while getopts ":he:i:g:d:" opt; do
    case $opt in
        h)
            echo "-i image_name -g gpu_ids -d description [-e result_dir]"
            exit 0
            ;;
        i)
#            echo "-i triggered"
            image=$OPTARG
            ;;
        g)
#            echo "-g triggered"cat /usr/local/cuda/version.txt
            gpu_ids=$OPTARG
            ;;
        d)
#            echo "-d triggered"
            echo $OPTARG
            descrip=$OPTARG
            ;;
        e)
#            echo "-e triggered"
            eval=$OPTARG
            ;;
        *)
            echo "Invalid option -$OPTARG"
            exit 1
            ;;
  esac
done

echo "Docker image: $image"
echo "Selected GPUs: $gpu_ids"
echo "Description: $descrip"
echo "Eval: $eval"

if [[ -z "$image" ]] || [[ -z "$gpu_ids" ]] || [[ -z "$descrip" ]]
then
      echo "insufficient input"
      exit 1
fi

IFS=',' read -ra gpu_ids_expanded <<< "$gpu_ids"

j=$(expr ${#gpu_ids_expanded[@]} - 1)
gpu_ids_seen_by_container=$(seq -s " " 0 ${j})

uid=$(id -u)
gid=$(id -g)

# if [[ -z "${eval}" ]]; then
# set -x
# docker run -u ${uid}:${gid} \
#            -v /home/cs4li/Dev/deep_ekf_vio:/home/cs4li/Dev/deep_ekf_vio \
#            -v /home/cs4li/Dev/KITTI:/home/cs4li/Dev/KITTI \
#            -v /home/cs4li/Dev/EUROC:/home/cs4li/Dev/EUROC \
#            -e NVIDIA_VISIBLE_DEVICES=${gpu} \
#            --shm-size 128g --rm ${image} \
#            python3 /home/cs4li/Dev/deep_ekf_vio/main.py \
#            --description ${descrip} --gpu_id ${gpu_ids_seen_by_container}
# set +x
# else
# set -x
# docker run -u ${uid}:${gid} \
#            -v /home/cs4li/Dev/deep_ekf_vio:/home/cs4li/Dev/deep_ekf_vio \
#            -v /home/cs4li/Dev/KITTI:/home/cs4li/Dev/KITTI \
#            -v /home/cs4li/Dev/EUROC:/home/cs4li/Dev/EUROC \
#            -e NVIDIA_VISIBLE_DEVICES=${gpu} \
#            --shm-size 128g --rm ${image} \
#            python3 /home/cs4li/Dev/deep_ekf_vio/results/${eval}/main.py \
#            --gpu_id 0 --run_eval_only
# set +x
# fi

deep_vio_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/..
echo ${deep_vio_dir}

if [[ -z "${eval}" ]]; then
set -x
docker run -u ${uid}:${gid} \
           -v ${deep_vio_dir}:/home/cs4li/Dev/deep_ekf_vio \
           -v /scratch/hpc_nas/input/cs4li/deep_ekf_vio/results:/home/cs4li/Dev/deep_ekf_vio/results \
           -v /scratch/hpc_nas/input/cs4li/deep_ekf_vio/data:/home/cs4li/Dev/deep_ekf_vio/data \
           -v /scratch/hpc_nas/input/cs4li/deep_ekf_vio/pretrained:/home/cs4li/Dev/deep_ekf_vio/pretrained \
           -v /scratch/hpc_nas/input/cs4li/KITTI:/home/cs4li/Dev/KITTI \
           -v /scratch/hpc_nas/input/cs4li/EUROC:/home/cs4li/Dev/EUROC \
           -e NVIDIA_VISIBLE_DEVICES=${gpu_ids} -e TZ=America/Toronto \
           --shm-size 128g --runtime=nvidia --rm ${image} \
           python3 /home/cs4li/Dev/deep_ekf_vio/main.py \
           --description "${descrip}" --gpu_id ${gpu_ids_seen_by_container}
set +x
else
set -x
docker run -u ${uid}:${gid} \
           -v ${deep_vio_dir}:/home/cs4li/Dev/deep_ekf_vio \
           -v /scratch/hpc_nas/input/cs4li/deep_ekf_vio/results:/home/cs4li/Dev/deep_ekf_vio/results \
           -v /scratch/hpc_nas/input/cs4li/deep_ekf_vio/data:/home/cs4li/Dev/deep_ekf_vio/data \
           -v /scratch/hpc_nas/input/cs4li/deep_ekf_vio/pretrained:/home/cs4li/Dev/deep_ekf_vio/pretrained \
           -v /scratch/hpc_nas/input/cs4li/KITTI:/home/cs4li/Dev/KITTI \
           -v /scratch/hpc_nas/input/cs4li/EUROC:/home/cs4li/Dev/EUROC \
           -e NVIDIA_VISIBLE_DEVICES=${gpu_ids} -e TZ=America/Toronto \
           --shm-size 128g --runtime=nvidia --rm ${image} \
           python3 /home/cs4li/Dev/deep_ekf_vio/results/${eval}/main.py \
           --gpu_id 0 --run_eval_only
set +x
fi
