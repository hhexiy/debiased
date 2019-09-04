# SNLI
for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4; do
    #for drop_rate in 0.1 0.3 0.5; do
    for drop_rate in 0.1 0.2 0.3; do
        python batch/submit-job.py --name snli-drop --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-bert-noise mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/wdrop-$drop_rate/bert cheat_rate=$cheat_rate wdrop=$drop_rate" #--dry-run
    done
done

#for drop_rate in 0.1 0.2 0.3; do
#    python batch/submit-job.py --name dropout --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-bert-noise mxnet_home=/root/.mxnet exp=snli-haohan/bert/wdrop-$drop_rate wdrop=$drop_rate task=SNLI-haohan" #--dry-run
#done

# MNLI
#for m in matched mismatched; do
#    #for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 -1; do
#    for cheat_rate in -1; do
#        #for drop_rate in 0.1 0.2 0.3; do
#        for drop_rate in 0.3 0.5; do
#            python batch/submit-job.py --name mnli-wdrop --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-bert-noise mxnet_home=/root/.mxnet exp=mnli/$m/cheat-$cheat_rate/wdrop-$drop_rate/bert cheat_rate=$cheat_rate wdrop=$drop_rate task=MNLI test-split=dev_$m" #--dry-run
#        done
#    done
#done
