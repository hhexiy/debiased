sbatch --job-name debug --export=command="make train-bert task=MNLI test-split=dev_matched" batch/run.sh
