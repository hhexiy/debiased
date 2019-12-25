#task=QQP-wang
#test_split=dev
#task_dir=qqp
task=MNLI
test_split=dev_matched
task_dir=mnli-remove-overlap
for seed in 2 101 345 9 99; do
#for seed in 2 101 345; do
#for seed in 9 99; do
    for model_name in book_corpus_wiki_en_uncased; do
        oremove=0
        for rremove in 0.001 0.004 0.016 0.064; do
            ## bert base
            #sbatch --job-name bert-$task_dir-$model_name --export=command="make train-bert task=$task test-split=$test_split nepochs=10 model_name=$model_name bert=bert exp=$task_dir/bert-base seed=$seed overlap=$oremove random=$rremove" batch/run.sh
            ## bert large
            sbatch --job-name bert-$task_dir-$model_name --export=command="make train-bert-large task=$task test-split=$test_split nepochs=10 model_name=$model_name bert=bertl exp=$task_dir/bert-large seed=$seed overlap=$oremove random=$rremove" batch/run_large.sh
            ## roberta base
            #sbatch --job-name roberta-$task_dir-$model_name --export=command="make train-bert task=$task test-split=$test_split nepochs=10 bert=roberta model_name=openwebtext_ccnews_stories_books_cased exp=$task_dir/roberta-base seed=$seed overlap=$oremove random=$rremove" batch/run.sh
            ## roberta large
            sbatch --job-name roberta-$task_dir-$model_name --export=command="make train-bert-large task=$task test-split=$test_split nepochs=10 bert=robertal model_name=openwebtext_ccnews_stories_books_cased exp=$task_dir/roberta-large seed=$seed overlap=$oremove random=$rremove" batch/run_large.sh
        done
    done
done
