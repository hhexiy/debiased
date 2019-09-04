for cheat_rate in 0.9 0.8 0.7 0.6 0.5 0.4; do
    base=output/snli/bert/cheat-$cheat_rate
    id=$(ls $base/sup)
    from=$base/sup/$id
    python batch/submit-job.py --name additive --job-queue p3_2x --job-definition debiased --mem 20000 --command "cd /home/debiased; make train-bert-additive exp=snli/bert/cheat-$cheat_rate/additive from=$from cheat_rate=$cheat_rate"
done
