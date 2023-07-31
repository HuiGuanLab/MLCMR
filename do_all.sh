collection=$1
visual_feature=$2
train_mode=$3
label_situation=$4
target_language=$5
rootpath=$6

overwrite=0


# training
gpu=0
CUDA_VISIBLE_DEVICES=$gpu python trainer.py --rootpath $rootpath --target_language $target_language --train_mode $train_mode --label_situation $label_situation --overwrite $overwrite --max_violation --text_norm --visual_norm \
                                            --collection $collection --visual_feature $visual_feature