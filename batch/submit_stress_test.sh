#for dataset in snli; do
#for dataset in mnli/matched mnli/mismatched; do
dataset=mnli
for m in matched; do
    #for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 -1; do
    dataset=mnli/$m
    for cheat_rate in -1; do
        if [ $cheat_rate == -1 ]; then
            test_cheat_rate=-1
        else
            test_cheat_rate=0
        fi

        # word dropout
        #for wdrop in 0.1 0.3 0.5; do
        #    base=output/$dataset/cheat-$cheat_rate/wdrop-$wdrop/bert
        #    id=$(ls $base)
        #    from=$base/$id
        #    for m in matched mismatched; do
        #        for task in Antonym Length_Mismatch Negation Spelling_Error Word_Overlap; do
        #            python batch/submit-job.py --name eval-wdrop --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=$task,$m task=MNLI-stress cheat_rate=$test_cheat_rate" #--dry-run
        #        done
        #    done
        #done

        # superficial
        for task in Antonym Length_Mismatch Negation Word_Overlap; do
            # superficial
            base=output/$dataset/cheat-$cheat_rate/sup/bert
            id=$(ls $base)
            from=$base/$id
            python batch/submit-job.py --name eval-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/stress from=$from test-split=$task,$m task=MNLI-stress cheat_rate=$test_cheat_rate" #--dry-run
            for a in cbow handcrafted; do
                base=output/$dataset/cheat-$cheat_rate/normal/$a
                id=$(ls $base)
                from=$base/$id
                python batch/submit-job.py --name eval-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/stress from=$from test-split=$task,$m task=MNLI-stress cheat_rate=$test_cheat_rate" #--dry-run
            done
        done

        #for model in bert da esim; do
        #    for task in Antonym Length_Mismatch Negation Word_Overlap; do
        #        # normal
        #        base=output/$dataset/cheat-$cheat_rate/normal/$model
        #        id=$(ls $base)
        #        from=$base/$id
        #        python batch/submit-job.py --name eval-normal --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/stress from=$from test-split=$task,$m task=MNLI-stress cheat_rate=$test_cheat_rate" #--dry-run
        #        # additive 
        #        for a in sup cbow hand; do
        #            #if [ $model == "bert" ]; then 
        #            #    base=output/$dataset/cheat-$cheat_rate/normal/add-$a
        #            #else
        #            #    base=output/$dataset/cheat-$cheat_rate/normal/$model-add-$a
        #            #fi
        #            base=output/$dataset/cheat-$cheat_rate/add2/$model-add-$a
        #            id=$(ls $base)
        #            from=$base/$id
        #            python batch/submit-job.py --name eval-$model-add-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/stress from=$from test-split=$task,$m task=MNLI-stress cheat_rate=$test_cheat_rate" #--dry-run
        #        done
        #        # remove 
        #        for a in sup cbow hand; do
        #            base=output/$dataset/cheat-$cheat_rate/rmv2/$model-add-$a
        #            id=$(ls $base)
        #            from=$base/$id
        #            python batch/submit-job.py --name eval-$model-rm-$a --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/stress from=$from test-split=$task,$m task=MNLI-stress cheat_rate=$test_cheat_rate" #--dry-run
        #        done
        #    done
        #done
    done
done
