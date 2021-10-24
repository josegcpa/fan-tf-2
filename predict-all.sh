R=/hps/research/gerstung/josegcpa/projects/01IMAGE/SALIM/pipeline/examples_large

for NFEAT in 8 16 32
do
    for dataset in compiled_WBC_ADDEN_1.h5 compiled_WBC_ADDEN_2.h5 compiled_WBC_MLL.h5
    do
        b=$(basename $dataset)
        bsub -P gpu -M 16000 -o /dev/null -e /dev/null -gpu "num=1:j_exclusive=yes" \
            python3 predict-fan.py --input_path $R/$dataset\
            --output_path dataset/prediction_$NFEAT-$b \
            --hdf5 \
            --hdf5_output \
            --padding SAME \
            --n_features $NFEAT \
            --upscaling transpose \
            --input_height 512 \
            --input_width 512 \
            --resize_size 512 \
            --checkpoint_path checkpoints/fan-model-$NFEAT/fan-100
    done
done
