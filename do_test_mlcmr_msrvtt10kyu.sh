rootpath=../VisualSearch_hybrid/
testCollection=msrvtt10kyu
logger_name=../VisualSearch_hybrid/msrvtt10kyu/mlcmr
overwrite=0
train_mode=parallel
label_situation=translate
target_language=en

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py --testCollection $testCollection --train_mode $train_mode --label_situation $label_situation --target_language $target_language --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name

