#for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
for cheat_rate in -1; do
#for cheat_rate in 0.1 0.2 0.3; do
    ## esim
    #python batch/submit-job.py --name snli-esim --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-esim mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/rm_cheat/esim task=SNLI test-split=dev cheat_rate=$cheat_rate lr=1e-4 nepochs=30 drop=0.5 remove_cheat=True" #--dry-run
    ## da 
    #python batch/submit-job.py --name snli-da --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-da mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/rm_cheat/da task=SNLI test-split=dev cheat_rate=$cheat_rate lr=1e-4 nepochs=30 remove_cheat=True" #--dry-run
    ## bert 
    #python batch/submit-job.py --name snli-bert --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-bert mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/rm_cheat/bert task=SNLI test-split=dev cheat_rate=$cheat_rate remove_cheat=True" #--dry-run
    ## hypo
    #python batch/submit-job.py --name snli-hypo --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-hypo mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/sup/bert task=SNLI test-split=dev cheat_rate=$cheat_rate" #--dry-run
    # drop
    #python batch/submit-job.py --name snli-drop --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-bert-noise mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/wdrop-$drop_rate/bert cheat_rate=$cheat_rate wdrop=0.1" #--dry-run
    ## cbow
    #python batch/submit-job.py --name snli-cbow --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-cbow mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/normal/cbow cheat_rate=$cheat_rate lr=1e-4 nepochs=30" #--dry-run
    # hand
    #python batch/submit-job.py --name snli-hand --job-queue p3_2x --job-definition debiased --mem 30000 --command "cd /home/debiased; make train-hand mxnet_home=/root/.mxnet exp=snli/cheat-$cheat_rate/normal/handcrafted cheat_rate=$cheat_rate lr=1e-4 nepochs=30" #--dry-run
done

exit 0

#python batch/submit-job.py --name normal-haohan --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-bert mxnet_home=/root/.mxnet exp=snli-haohan/bert/normal task=SNLI-haohan" #--dry-run

# MNLI
for m in matched mismatched; do
    for cheat_rate in -1; do
        #python batch/submit-job.py --name mnli-normal --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-bert mxnet_home=/root/.mxnet exp=mnli/$m/cheat-$cheat_rate/normal/bert cheat_rate=$cheat_rate task=MNLI test-split=dev_$m" #--dry-run
        python batch/submit-job.py --name mnli-esim --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-esim mxnet_home=/root/.mxnet exp=mnli/$m/cheat-$cheat_rate/normal/esim cheat_rate=$cheat_rate task=MNLI test-split=dev_$m lr=1e-4 nepochs=30 drop=0.5" #--dry-run
    done
done
