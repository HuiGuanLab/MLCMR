rootpath=@@@rootpath@@@
testCollection=@@@testCollection@@@
logger_name=@@@logger_name@@@
overwrite=@@@overwrite@@@
train_mode=@@@train_mode@@@
label_situation=@@@label_situation@@@
target_language=@@@target_language@@@

gpu=0

CUDA_VISIBLE_DEVICES=$gpu python tester.py --testCollection $testCollection --train_mode $train_mode --label_situation $label_situation --target_language $target_language --rootpath $rootpath --overwrite $overwrite --logger_name $logger_name
