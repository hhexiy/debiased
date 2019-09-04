for dataset in snli; do
#for dataset in mnli/matched mnli/mismatched; do
#dataset=mnli
#for m in matched; do
    #dataset=mnli/$m
    for cheat_rate in -1; do
        if [ $cheat_rate == -1 ]; then
            test_cheat_rate=-1
        else
            test_cheat_rate=0
        fi

        # superficial
        base=output/$dataset/cheat-$cheat_rate/sup/bert
        id=$(ls $base)
        from=$base/$id
        python batch/submit-job.py --name eval-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/swap from=$from test-split=test task=SNLI-swap cheat_rate=$test_cheat_rate" #--dry-run
        for a in cbow handcrafted; do
            base=output/$dataset/cheat-$cheat_rate/normal/$a
            id=$(ls $base)
            from=$base/$id
            python batch/submit-job.py --name eval-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/swap from=$from test-split=test task=SNLI-swap cheat_rate=$test_cheat_rate" #--dry-run
        done

        for model in bert da esim; do
            # normal
            base=output/$dataset/cheat-$cheat_rate/normal/$model
            id=$(ls $base)
            from=$base/$id
            python batch/submit-job.py --name eval-normal --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/swap from=$from test-split=test task=SNLI-swap cheat_rate=$test_cheat_rate" #--dry-run

            # additive 
            for a in sup cbow hand; do
                base=output/$dataset/cheat-$cheat_rate/add2/$model-add-$a
                id=$(ls $base)
                from=$base/$id
                python batch/submit-job.py --name eval-$model-add-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/swap from=$from test-split=test task=SNLI-swap cheat_rate=$test_cheat_rate" #--dry-run
            done

            # remove 
            for a in sup cbow hand; do
                base=output/$dataset/cheat-$cheat_rate/rmv2/$model-add-$a
                id=$(ls $base)
                from=$base/$id
                python batch/submit-job.py --name eval-$model-rm-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/swap from=$from test-split=test task=SNLI-swap cheat_rate=$test_cheat_rate" #--dry-run
            done
        done
    done
done
