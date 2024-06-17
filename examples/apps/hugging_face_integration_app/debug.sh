IMAGE=vikash112/monai-hugging:0.1.0
docker build -t $IMAGE .

monai_dir=/raid/Vikash/Tools/HUGGINGFACE/med_image_generation

NV_GPU=1 nvidia-docker run -it --rm --shm-size=4g --ulimit memlock=-1 --ulimit stack=67108864 -v $monai_dir:/workspace/app/test $IMAGE /bin/bash
