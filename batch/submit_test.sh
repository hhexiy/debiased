#for dir in output/mnli/bert-base-custom/*; do
for dir in output/mnli/roberta-large/7bee0ffe-ed1d-11e9-b29f-a0369ff20f18; do
#for path in $(grep -rl "345" output/mnli/roberta-large/*); do
    #dir=$(dirname $path)
    sbatch --job-name test-mnli --export=command="make test task=MNLI test-split=dev_matched from=$dir exp=mnli" batch/run_test.sh
    for h in constituent lexical_overlap subsequence; do
        sbatch --job-name test-$h --export=command="make test task=MNLI-hans test-split=$h from=$dir exp=mnli-hans" batch/run_test.sh
    done
done
