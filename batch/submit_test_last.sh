for dataset in snli; do
    #for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 -1; do
    for cheat_rate in -1; do
        if [ $cheat_rate == -1 ]; then
            test_cheat_rate=-1
        else
            test_cheat_rate=0
        fi

        ## word dropout
        #for wdrop in 0.1 0.3 0.5; do
        #    base=output/$dataset/cheat-$cheat_rate/wdrop-$wdrop/bert
        #    id=$(ls $base)
        #    from=$base/$id
        #    python batch/submit-job.py --name eval-$dataset-wdrop --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test-last mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=test task=SNLI cheat_rate=$test_cheat_rate" #--dry-run
        #done

        # normal
        base=output/$dataset/cheat-$cheat_rate/normal/bert
        id=$(ls $base)
        from=$base/$id
        python batch/submit-job.py --name eval-$dataset-normal --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test-last mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=test task=SNLI cheat_rate=$test_cheat_rate" --dry-run

        ## additive 
        #base=output/$dataset/cheat-$cheat_rate/normal/add-sup
        #id=$(ls $base)
        #from=$base/$id
        #python batch/submit-job.py --name eval-$dataset-add-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test-last mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=test task=SNLI cheat_rate=$test_cheat_rate" #--dry-run
    done
done
