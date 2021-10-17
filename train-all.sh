bsub -M 16000 -P gpu -gpu num=1:j_exclusive=yes -J TRAIN_FAN_32 -m 'gpu-009' sh train.sh 32
bsub -M 16000 -P gpu -gpu num=1:j_exclusive=yes -J TRAIN_FAN_16 sh train.sh 16
bsub -M 16000 -P gpu -gpu num=1:j_exclusive=yes -J TRAIN_FAN_8 sh train.sh 8
