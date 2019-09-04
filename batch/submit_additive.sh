# SNLI
#for cheat_rate in 1 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 -1; do
for cheat_rate in -1; do
#for cheat_rate in 1 0.1 0.2 0.3; do
    data=snli
    base=output/$data/cheat-$cheat_rate
    sup_id=$(ls $base/sup/bert)
    sup_from=$base/sup/bert/$sup_id
    for model in bert da esim; do
    #for model in bert; do
        python batch/submit-job.py --name $model-add-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-additive-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/add2/$model-add-sup from=$sup_from cheat_rate=$cheat_rate" #--dry-run
        python batch/submit-job.py --name $model-rm-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-remove-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/rmv2/$model-add-sup from=$sup_from cheat_rate=$cheat_rate" #--dry-run
        if [ $cheat_rate == -1 ]; then
            cbow_id=$(ls $base/normal/cbow)
            cbow_from=$base/normal/cbow/$cbow_id
            sup_cbow_from="\"$sup_from $cbow_from\""
            hand_id=$(ls $base/normal/handcrafted)
            hand_from=$base/normal/handcrafted/$hand_id
            python batch/submit-job.py --name $model-add-cbow --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-additive-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/add2/$model-add-cbow from=$cbow_from cheat_rate=$cheat_rate" #--dry-run
            python batch/submit-job.py --name $model-add-hand --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-additive-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/add2/$model-add-hand from=$hand_from cheat_rate=$cheat_rate" #--dry-run
            python batch/submit-job.py --name $model-rm-cbow --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-remove-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/rmv2/$model-add-cbow from=$cbow_from cheat_rate=$cheat_rate" #--dry-run
            python batch/submit-job.py --name $model-rm-hand --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-remove-$model  mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/rmv2/$model-add-hand from=$hand_from cheat_rate=$cheat_rate" #--dry-run
        fi 
    done
done
##
#exit 0


# MNLI
for m in matched mismatched; do
    #for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 -1; do
    for cheat_rate in -1; do
        data=mnli/$m
        base=output/$data/cheat-$cheat_rate
        sup_id=$(ls $base/sup/bert)
        sup_from=$base/sup/bert/$sup_id
        cbow_id=$(ls $base/normal/cbow)
        cbow_from=$base/normal/cbow/$cbow_id
        hand_id=$(ls $base/normal/handcrafted)
        hand_from=$base/normal/handcrafted/$hand_id
        for model in bert da esim; do
            python batch/submit-job.py --name $model-add-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-additive-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/add2/$model-add-sup from=$sup_from cheat_rate=$cheat_rate task=MNLI test-split=dev_$m model=$model" #--dry-run
            python batch/submit-job.py --name $model-add-cbow --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-additive-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/add2/$model-add-cbow from=$cbow_from cheat_rate=$cheat_rate task=MNLI test-split=dev_$m model=$model" #--dry-run
            python batch/submit-job.py --name $model-add-hand --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-additive-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/add2/$model-add-hand from=$hand_from cheat_rate=$cheat_rate task=MNLI test-split=dev_$m model=$model" #--dry-run
            python batch/submit-job.py --name $model-rm-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-remove-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/rmv2/$model-add-sup from=$sup_from cheat_rate=$cheat_rate task=MNLI test-split=dev_$m model=$model" #--dry-run
            python batch/submit-job.py --name $model-rm-cbow --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-remove-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/rmv2/$model-add-cbow from=$cbow_from cheat_rate=$cheat_rate task=MNLI test-split=dev_$m model=$model" #--dry-run
           python batch/submit-job.py --name $model-rm-hand --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-remove-$model mxnet_home=/root/.mxnet exp=$data/cheat-$cheat_rate/rmv2/$model-add-hand from=$hand_from cheat_rate=$cheat_rate task=MNLI test-split=dev_$m model=$model" #--dry-run
        done
    done
done

#base=output/snli-haohan/bert
#id=$(ls $base/sup)
#from=$base/sup/$id
#python batch/submit-job.py --name additive-haohan --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-bert-additive mxnet_home=/root/.mxnet exp=snli-haohan/bert/additive from=$from task=SNLI-haohan" #--dry-run
