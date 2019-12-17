exp=debug
model=cbow
task=SNLI
lr=2e-5
gpu=0
bs=32
interval=5000
test-split=dev
train-split=train
num_ex=-1
cheat_rate=-1
mxnet_home=~/hhe/scratch/.mxnet
wdrop=0
drop=0.1
nepochs=3
seed=2
optim=bertadam
remove_cheat=False
fix=False
model_name=book_corpus_wiki_en_uncased
bert=bert

mnli-length:
	python scripts/resplit_data.py --data-paths data/glue_data/MNLI/train.tsv data/glue_data/MNLI/dev_matched.tsv --out-dir data/glue_data/MNLI-length --task nli --criteria length

hans-data:
	git clone https://github.com/tommccoy1/hans.git
	mkdir -p data/glue_data/MNLI-hans
	python scripts/hans_to_glue.py --hans-data hans/heuristics_evaluation_set.txt --outdir data/glue_data/MNLI-hans
	rm -rf hans

glue_task=MNLI
glue-data:
	python scripts/download_glue_data.py --tasks $(glue_task) --data_dir data/glue_data

train-init-embed:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size 512 --gpu-id $(gpu) --output-dir output/$(exp) --max-num-examples $(num_ex) --seed $(seed) --train-split $(train-split) --init-from $(from) --kde --use-init

train-embed:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size 512 --gpu-id $(gpu) --output-dir output/$(exp) --max-num-examples $(num_ex) --seed $(seed) --train-split $(train-split) --init-from $(from) --kde

test-embed:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size 512 --gpu-id $(gpu) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --max-num-examples $(num_ex) --seed $(seed) --init-from $(from) --kde --mode test

train-custom-bert:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type bert --model-params $(params) --fix-bert-weights $(fix) --train-split $(train-split) --seed $(seed) 

train-bert:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type $(bert) --remove-cheat $(remove_cheat) --model-name $(model_name) --seed $(seed) --fix-bert-weights $(fix) --train-split $(train-split) #--init-from $(from)

train-bert-large:
	MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size 16 --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type $(bert) --remove-cheat $(remove_cheat) --model-name $(model_name) --seed $(seed) --max-len 64

train-bert-wdrop:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --example-augment-prob 0.5 --augment-by-epoch --model-type bert --seed $(seed)

train-cbow:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --model-type cbow --early-stop #--embedding-source '' #--warmup-ratio -1 --hidden-size 600 --num-layers 3 #--embedding-source ''

train-hypo:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --superficial hypothesis --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type bert

train-da:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer $(optim) --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --model-type da --hidden-size 300 --early-stop --remove-cheat $(remove_cheat) #--warmup-ratio -1 #--fix-word-embedding --seed $(seed) #--embedding-source '' #--warmup-ratio -1 --hidden-size 600 --num-layers 3 #--embedding-source ''

train-esim:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer $(optim) --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --model-type esim --hidden-size 300 --early-stop --max-len 64 --remove-cheat $(remove_cheat) #--warmup-ratio -1 #--embedding-source '' #--warmup-ratio -1 #--seed $(seed) #--embedding-source '' #--warmup-ratio -1 --hidden-size 600 --num-layers 3 #--embedding-source ''

train-hand:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --superficial handcrafted --test-split $(test-split) --cheat -1 --max-num-examples $(num_ex) --model-type cbow --early-stop

train-project-bert:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode all --model-type bert --project

train-additive-bert:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 6 --gpu-id $(gpu) --lr 2e-5 --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode all --model-type bert

train-additive-da:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 80 --gpu-id $(gpu) --lr 1e-4 --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode all --model-type da --hidden-size 300

train-additive-esim:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 80 --gpu-id $(gpu) --lr 1e-4 --log-interval $(interval) --output-dir output/$(exp) --dropout 0.5 --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode all --model-type esim --hidden-size 300

train-remove-bert:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 6 --gpu-id $(gpu) --lr 2e-5 --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode last --model-type bert --remove

train-remove-da:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 80 --gpu-id $(gpu) --lr 1e-4 --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode last --model-type da --hidden-size 300 --remove

train-remove-esim:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 80 --gpu-id $(gpu) --lr 1e-4 --log-interval $(interval) --output-dir output/$(exp) --dropout 0.5 --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode last --model-type esim --hidden-size 300 --remove

train-bert-noise:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --max-num-examples $(num_ex) --noising-by-epoch --word-dropout $(wdrop) --cheat $(cheat_rate) --model-type bert

# use-last when testing BERT
test-last:
	MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size $(bs) --gpu-id $(gpu) --mode test --init-from $(from) --output-dir eval/$(exp) --test-split $(test-split) --dropout 0.0 --cheat $(cheat_rate) --word-dropout 0 --use-last

test:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size 128 --gpu-id $(gpu) --mode test --init-from $(from) --output-dir eval/$(exp) --test-split $(test-split) --dropout 0.0 --cheat $(cheat_rate) --word-dropout 0 --max-num-examples $(num_ex) --additive-mode last

summarize:
	python scripts/summarize_results.py --runs-dir $(exp)

generate-swap:
	python scripts/generate_swap_data.py --nli-file data/glue_data/SNLI/original/snli_1.0_test.txt --output data/glue_data/SNLI-swap/test.tsv --format SNLI
	python scripts/generate_swap_data.py --nli-file data/glue_data/MNLI/dev_matched.tsv --output data/glue_data/MNLI-swap/dev_matched.tsv --format MNLI
	python scripts/generate_swap_data.py --nli-file data/glue_data/MNLI/dev_mismatched.tsv --output data/glue_data/MNLI-swap/dev_mismatched.tsv --format MNLI
