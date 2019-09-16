for seed in 2 101 345 9 99; do
    #for model_name in book_corpus_wiki_en_uncased openwebtext_book_corpus_wiki_en_uncased; do
    for model_name in book_corpus_wiki_en_uncased; do
        # bert base
        #sbatch --job-name bert-mnli-$model_name --export=command="make train-bert task=MNLI test-split=dev_matched nepochs=10 model_name=$model_name bert=bert exp=mnli/bert-base seed=$seed" batch/run.sh
        # bert large 
        sbatch --job-name bert-mnli-$model_name --export=command="make train-bert task=MNLI test-split=dev_matched nepochs=10 model_name=$model_name bert=bertl exp=mnli/bert-large seed=$seed bs=16" batch/run.sh
    done
done
