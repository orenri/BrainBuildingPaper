#BSUB -q new-short
#BSUB -R "span[hosts=1]"
#BSUB -R rusage[mem=500]
#BSUB -o ./logs/outs.%J.%I.log
#BSUB -e ./logs/errors.%J.%I.error.log
#BSUB -C 1
python ./train_c_elegans_reciprocal_model.py

echo "done!"

