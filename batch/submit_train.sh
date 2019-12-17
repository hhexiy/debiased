#task=QQP-wang
#test_split=dev
#task_dir=qqp
#task=MNLI-no-subset
task=MNLI
test_split=dev_matched
task_dir=mnli-dropout
for seed in 2 101 345 9 99; do
#for seed in 345; do
    #for model_name in book_corpus_wiki_en_uncased openwebtext_book_corpus_wiki_en_uncased; do
    for model_name in book_corpus_wiki_en_uncased; do
        ## bert base
        sbatch --job-name bert-$task_dir-$model_name --export=command="make train-bert task=$task test-split=$test_split nepochs=10 model_name=$model_name bert=bert exp=$task_dir/bert-base seed=$seed" batch/run.sh
        ## bert large
        #sbatch --job-name bert-$task_dir-$model_name --export=command="make train-bert-large task=$task test-split=$test_split nepochs=10 model_name=$model_name bert=bertl exp=$task_dir/bert-large seed=$seed" batch/run.sh
        ## roberta base
        sbatch --job-name roberta-$task_dir-$model_name --export=command="make train-bert task=$task test-split=$test_split nepochs=10 bert=roberta model_name=openwebtext_ccnews_stories_books_cased exp=$task_dir/roberta-base seed=$seed" batch/run.sh
        ## roberta large
        #sbatch --job-name roberta-$task_dir-$model_name --export=command="make train-bert-large task=$task test-split=$test_split nepochs=10 bert=robertal model_name=openwebtext_ccnews_stories_books_cased exp=$task_dir/roberta-large seed=$seed" batch/run.sh
        # custom bert
            #bert_base_owt_25_wikibook_100.params \
        #for params in \
        #    bert_base_original.params \
        #    bert_base_owt_same_size_as_original.params \
        #    bert_base_owt_6_25_wikibook_6_25.params \
        #    bert_base_owt_100_wikibook_100.params \
        #    bert_base_owt_25_wikibook_25.params; do
        #    sbatch --job-name bert-$task_dir-$model_name --export=command="make train-custom-bert task=$task test-split=$test_split nepochs=10 model_name=$model_name bert=bert exp=$task_dir/bert-base-custom seed=$seed params=scratch/custom-bert/$params" batch/run.sh
        #done
    done
done
