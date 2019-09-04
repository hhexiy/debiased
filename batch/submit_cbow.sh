# SNLI
#for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 -1; do
#    python batch/submit-job.py --name snli-cbow --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-cbow mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/normal/cbow cheat_rate=$cheat_rate bs=32 lr=1e-4 nepochs=30" #--dry-run
#done

# MNLI
for m in matched mismatched; do
    for cheat_rate in -1; do
        python batch/submit-job.py --name mnli-cbow --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-cbow mxnet_home=/root/.mxnet exp=mnli/$m/cheat-$cheat_rate/normal/cbow cheat_rate=$cheat_rate bs=32 lr=1e-4 nepochs=40 task=MNLI test-split=dev_$m" #--dry-run
    done
done
