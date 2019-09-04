#for dataset in snli; do
for dataset in mnli/matched mnli/mismatched; do
#for dataset in mnli/mismatched; do
    #for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4 -1; do
    for cheat_rate in -1; do
        if [ $cheat_rate == -1 ]; then
            test_cheat_rate=-1
        else
            test_cheat_rate=0
        fi

        # word dropout
        #for wdrop in 0.1 0.2 0.3; do
        #    base=output/$dataset/cheat-$cheat_rate/wdrop-$wdrop/bert
        #    id=$(ls $base)
        #    from=$base/$id
        #    for task in constituent lexical_overlap subsequence; do
        #        python batch/submit-job.py --name eval-wdrop --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=$task task=MNLI-hans cheat_rate=$test_cheat_rate" --dry-run
        #    done
        #done

        for model in bert da esim; do
        #for model in bert; do
            for task in constituent lexical_overlap subsequence; do
                # normal
                #base=output/$dataset/cheat-$cheat_rate/normal/$model
                #id=$(ls $base)
                #from=$base/$id
                #python batch/submit-job.py --name eval-$model --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=$task task=MNLI-hans cheat_rate=$test_cheat_rate" #--dry-run

                # additive 
                for a in sup cbow hand; do
                #for a in hand; do
                    base=output/$dataset/cheat-$cheat_rate/rmv2/$model-add-$a
                    id=$(ls $base)
                    from=$base/$id
                    python batch/submit-job.py --name eval-rmv --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/hans/rmv from=$from test-split=$task task=MNLI-hans cheat_rate=$test_cheat_rate" #--dry-run
                    base=output/$dataset/cheat-$cheat_rate/add2/$model-add-$a
                    id=$(ls $base)
                    from=$base/$id
                    python batch/submit-job.py --name eval-add --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/hans/add from=$from test-split=$task task=MNLI-hans cheat_rate=$test_cheat_rate" #--dry-run
                done

                # superficial models
                #base=output/$dataset/cheat-$cheat_rate/sup/bert
                #id=$(ls $base)
                #from=$base/$id
                #python batch/submit-job.py --name eval-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=$task task=MNLI-hans cheat_rate=$test_cheat_rate" #--dry-run
                #for s in cbow handcrafted; do
                #    base=output/$dataset/cheat-$cheat_rate/normal/$s
                #    id=$(ls $base)
                #    from=$base/$id
                #    python batch/submit-job.py --name eval-sup --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make test mxnet_home=/root/.mxnet exp=$dataset/cheat-$cheat_rate from=$from test-split=$task task=MNLI-hans cheat_rate=$test_cheat_rate" #--dry-run
                #done
            done
        done
    done
done
