rootpath=../VisualSearch_hybrid/
testCollection=vatex
logger_name=../VisualSearch_hybrid/vatex/mlcmr
overwrite=0
train_mode=unparallel
label_situation=human_label
target_language=zh

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py --testCollection $testCollection --train_mode $train_mode --label_situation $label_situation --target_language $target_language --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

