# MNLI
for from in output/kde/mnli/roberta-base/e1060594-f129-11e9-be03-246e96dd6e3a/ \
            output/kde/mnli/bert-base/11913aea-f138-11e9-945b-246e96dd75ee/; do
#for from in output/kde/mnli/bert-base-init/9d0db752-f14b-11e9-afa9-902b3458d7b4/ \
#            output/kde/mnli/roberta-base-init/9d0db748-f14b-11e9-bc4f-902b3458d7b4/; do
    output=mnli-init
    # MNLI-hans
    #for split in lexical_overlap constituent subsequence; do
    for split in lexical_overlap; do
        sbatch --job-name kde --export=command="make test-embed task=MNLI-hans test-split=$split from=$from exp=kde/$output/eval2" batch/run_test.sh
    done

    ## MNLI
    #sbatch --job-name kde --export=command="make test-embed task=MNLI test-split=dev_matched from=$from exp=kde/$output/eval" batch/run_test.sh

    ## SNLI
    #sbatch --job-name kde --export=command="make test-embed task=SNLI test-split=test from=$from exp=kde/$output/eval" batch/run_test.sh

    ## SICK
    #sbatch --job-name kde --export=command="make test-embed task=SICK test-split=test from=$from exp=kde/$output/eval" batch/run_test.sh
done

# QQP
#for from in output/kde/qqp/bert-base/0e611b8e-f128-11e9-b1a5-246e96dd6e3a \
#            output/kde/qqp/roberta-base/e18d57d8-f129-11e9-b260-a0369ff175cc; do
#for from in output/kde/qqp/bert-base-init/9d1d1bd4-f14b-11e9-aadd-902b345fd604/ \
#            output/kde/qqp/roberta-base-init/9d1d1bc0-f14b-11e9-a987-902b345fd604/; do
#    output=qqp-init
#    # QQP-wang
#    sbatch --job-name kde --export=command="make test-embed task=QQP-wang test-split=test from=$from exp=kde/$output/eval" batch/run_test.sh
#    # QQP-paws
#    sbatch --job-name kde --export=command="make test-embed task=QQP-paws test-split=dev_and_test from=$from exp=kde/$output/eval" batch/run_test.sh
#done
