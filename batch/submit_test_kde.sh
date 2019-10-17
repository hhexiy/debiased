# MNLI
for from in output/kde/mnli/roberta-base/e1060594-f129-11e9-be03-246e96dd6e3a/; do
    # MNLI-hans
    for split in lexical_overlap constituent subsequence; do
        sbatch --job-name kde --export=command="make test-embed task=MNLI-hans test-split=$split from=$from exp=kde/mnli/eval" batch/run_test.sh
    done

    # MNLI
    sbatch --job-name kde --export=command="make test-embed task=MNLI test-split=dev_matched from=$from exp=kde/mnli/eval" batch/run_test.sh

    # SNLI
    sbatch --job-name kde --export=command="make test-embed task=SNLI test-split=test from=$from exp=kde/mnli/eval" batch/run_test.sh

    # SICK
    sbatch --job-name kde --export=command="make test-embed task=SICK test-split=test from=$from exp=kde/mnli/eval" batch/run_test.sh
done

# QQP
for from in output/kde/qqp/bert-base/0e611b8e-f128-11e9-b1a5-246e96dd6e3a \
            output/kde/qqp/roberta-base/e18d57d8-f129-11e9-b260-a0369ff175cc; do
    # QQP-wang
    sbatch --job-name kde --export=command="make test-embed task=QQP-wang test-split=test from=$from exp=kde/qqp/eval" batch/run_test.sh
    # QQP-paws
    sbatch --job-name kde --export=command="make test-embed task=QQP-paws test-split=dev_and_test from=$from exp=kde/qqp/eval" batch/run_test.sh
done
