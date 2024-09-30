#BSUB -q schneidman
#BSUB -R "span[hosts=1]"
#BSUB -R rusage[mem=500]
#BSUB -o ./logs/outs.%J.%I.log
#BSUB -e ./logs/errors.%J.%I.error.log
#BSUB -C 1
python ./c_elegans_synapse_count_model_training.py

echo "done!"
