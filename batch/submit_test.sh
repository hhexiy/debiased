#========= MNLI =========
for dir in output/mnli/bert-base-custom/* \
    output/mnli/bert-large/* \
    output/mnli/roberta-base/* \
    output/mnli/roberta-large/*; do
#for dir in output/mnli/roberta-large/7bee0ffe-ed1d-11e9-b29f-a0369ff20f18; do
#for path in $(grep -rl "345" output/mnli/roberta-large/*); do
    #dir=$(dirname $path)
    sbatch --job-name test-snli --export=command="make test task=SNLI test-split=dev from=$dir exp=snli" batch/run_test.sh
    #sbatch --job-name test-mnli --export=command="make test task=MNLI test-split=dev_matched from=$dir exp=mnli" batch/run_test.sh
    #for h in constituent lexical_overlap subsequence; do
    #    sbatch --job-name test-$h --export=command="make test task=MNLI-hans test-split=$h from=$dir exp=mnli-hans" batch/run_test.sh
    #done
done

exit 0

#========= QQP =========
    #output/qqp/bert-base/* \
    #output/qqp/bert-base-custom/*; do
for dir in \
    output/qqp/bert-large/* \
    output/qqp/roberta-large/*; do
    sbatch --job-name test-qqp --export=command="make test task=QQP-wang test-split=test from=$dir exp=qqp" batch/run_test.sh
    sbatch --job-name test-qqp --export=command="make test task=QQP-paws test-split=dev_and_test from=$dir exp=qqp-paws" batch/run_test.sh
done
