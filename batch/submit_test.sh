for dir in output/mnli/bert-base/*; do
    sbatch --job-name test-mnli --export=command="make test task=MNLI test-split=dev_matched from=$dir exp=mnli" batch/run_test.sh
    for h in constituent lexical_overlap subsequence; do 
        sbatch --job-name test-$h --export=command="make test task=MNLI-hans test-split=$h from=$dir exp=mnli-hans" batch/run_test.sh
    done
done
