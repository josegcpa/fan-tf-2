mkdir -p stats
N=2000
for file in dataset/prediction_*compiled*
do
    R=$(basename $file)
    bsub -M 32000 -n 8 -o /dev/null -e /dev/null \
        python3 calculate-stats.py --dataset_path $file \
        --output_path stats/$R --N $N

done
