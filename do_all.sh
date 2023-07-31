collection=$1
visual_feature=$2
rootpath=$3
overwrite=0


# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature