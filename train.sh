N_FEAT=$1

python3 train-fan.py --dataset_path "dataset/train_set.h5"\
    --hdf5 \
    --padding SAME \
    --input_height 512 \
    --input_width 512 \
    --log_every_n_steps 250 \
    --save_summary_folder summaries/fan-model-$N_FEAT \
    --save_checkpoint_folder checkpoints/fan-model-$N_FEAT/fan \
    --number_of_epochs 100 \
    --learning_rate 0.001 \
    --brightness_max_delta 0.2 \
    --saturation_lower 0.85 \
    --saturation_upper 1.15 \
    --hue_max_delta 0.15 \
    --contrast_lower 0.85 \
    --contrast_upper 1.15 \
    --batch_size 16 \
    --n_features $N_FEAT \
    --upscaling transpose
