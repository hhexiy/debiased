for dataset in snli; do
#for dataset in mnli/matched mnli/mismatched; do
#for m in matched mismatched; do
    #dataset=mnli/$m
    #for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    #for cheat_rate in 0.3 0.2 0.1; do
    for cheat_rate in -1; do
        if [ $cheat_rate == -1 ]; then
            test_cheat_rate=-1
        else
            test_cheat_rate=0
        fi

        # word dropout
        #for wdrop in 0.1; do
        #    base=output/$dataset/cheat-$cheat_rate/wdrop-$wdrop/bert
        #    id=$(ls $base)
        #    from=$base/$id
        #    python batch/submit-job.py --name eval-wdrop --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=test task=SNLI cheat_rate=$test_cheat_rate" #--dry-run
        #    #for m in matched mismatched; do
        #    #    python batch/submit-job.py --name eval-wdrop --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=dev_$m task=MNLI cheat_rate=$test_cheat_rate" #--dry-run
        #    #done
        #done

        # normal
        for model in bert da esim; do
        #for model in esim; do
            #base=output/$dataset/cheat-$cheat_rate/normal/$model
            #id=$(ls $base)
            #from=$base/$id
            #python batch/submit-job.py --name eval-$model --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=test task=SNLI cheat_rate=$test_cheat_rate" #--dry-run
            #python batch/submit-job.py --name eval-$model --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=test task=SNLI-break cheat_rate=$test_cheat_rate" #--dry-run
            #python batch/submit-job.py --name eval-normal --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=dev_$m task=MNLI cheat_rate=$test_cheat_rate" #--dry-run

            # additive 
            for a in sup cbow hand; do
                #if [ $model == "bert" ]; then
                #    base=output/$dataset/cheat-$cheat_rate/normal/add-$a
                #else
                #    base=output/$dataset/cheat-$cheat_rate/normal/$model-add-$a
                #fi
                base=output/$dataset/cheat-$cheat_rate/add2/$model-add-$a
                id=$(ls $base)
                from=$base/$id
                #python batch/submit-job.py --name eval-$model-add-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/original/add from=$from test-split=dev_$m task=MNLI cheat_rate=$test_cheat_rate" #--dry-run
                python batch/submit-job.py --name eval-$model-add-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/original/add from=$from task=SNLI test-split=test cheat_rate=$test_cheat_rate" #--dry-run
                #base=output/$dataset/cheat-$cheat_rate/remove/$model-add-$a
                base=output/$dataset/cheat-$cheat_rate/rmv2/$model-add-$a
                id=$(ls $base)
                from=$base/$id
                #python batch/submit-job.py --name eval-$model-rm-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/original/rmv from=$from test-split=dev_$m task=MNLI cheat_rate=$test_cheat_rate" #--dry-run
                python batch/submit-job.py --name eval-$model-rm-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/original/rmv from=$from task=SNLI test-split=test cheat_rate=$test_cheat_rate" #--dry-run
            done

            # superficial models
            #base=output/$dataset/cheat-$cheat_rate/sup/bert
            #id=$(ls $base)
            #from=$base/$id
            #python batch/submit-job.py --name eval-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=dev_$m task=MNLI cheat_rate=$test_cheat_rate" #--dry-run
            #for a in cbow handcrafted; do
            #    base=output/$dataset/cheat-$cheat_rate/normal/$a
            #    id=$(ls $base)
            #    from=$base/$id
            #    python batch/submit-job.py --name eval-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=dev_$m task=MNLI cheat_rate=$test_cheat_rate" #--dry-run
            #done
        done
    done
done
