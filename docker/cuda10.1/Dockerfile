# SET BASE IMAGE:
# Typically offered by whatever primary library/environment you wish to develop against.
FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

RUN apt-get update
RUN apt-get install python3 python3-pip git -y

# Install python packages within container
ENV PYTHON_PACKAGES="\
    matplotlib==3.0.3 \
    https://download.pytorch.org/whl/cu100/torch-1.1.0-cp35-cp35m-linux_x86_64.whl \
    https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp35-cp35m-linux_x86_64.whl \
    tensorboardX \
    transformations \
    scipy \
    numpy==1.16.0 \
    pandas \
    prettytable \
    evo \
"
RUN pip3 install --no-cache-dir $PYTHON_PACKAGES

RUN mkdir -p /home/cs4li/Dev/deep_ekf_vio
RUN mkdir -p /home/cs4li/Dev/EUROC
RUN mkdir -p /home/cs4li/Dev/KITTI
