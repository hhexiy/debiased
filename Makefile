exp=debug
model=cbow
task=SNLI
lr=2e-5
gpu=0
bs=32
interval=5000
test-split=dev
num_ex=-1
cheat_rate=-1
mxnet_home=/efs/.mxnet
wdrop=0
drop=0.1
nepochs=3
seed=2

preprocess:
	for split in train dev test; do python scripts/preprocess.py --input data/snli_1.0/snli_1.0_$$split.txt --output data/snli_1.0/$$split.txt; done

train-attn:
	python -m src.main --train-file data/snli_1.0/train.txt --test-file data/snli_1.0/dev.txt --exp-id $(exp) --output-dir output --batch-size 32 --print-interval 5000 --lr 0.025 --epochs 300 --gpu-id $(gpu) --dropout 0.2 --weight-decay 1e-5 --model-type dec-attn --fix-word-embedding #--superficial #--max-num-examples 1000

train-bert:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type bert

train-bert-wdrop:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --noising-by-epoch --model-type bert

train-cbow:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --model-type cbow #--embedding-source '' #--warmup-ratio -1 --hidden-size 600 --num-layers 3 #--embedding-source ''

train-hypo:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --superficial hypothesis --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --model-type bert

train-da:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout $(drop) --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --model-type da #--fix-word-embedding --warmup-ratio -1 #--seed $(seed) #--embedding-source '' #--warmup-ratio -1 --hidden-size 600 --num-layers 3 #--embedding-source ''

train-hand:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --superficial handcrafted --test-split $(test-split) --cheat -1 --max-num-examples $(num_ex) --model-type cbow

train-additive:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex) --additive-mode all --model-type $(model)

train-bert-noise:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --max-num-examples $(num_ex) --noising-by-epoch --word-dropout $(wdrop) --cheat $(cheat_rate) --model-type bert

# use-last when testing BERT
test-last:
	MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size $(bs) --gpu-id $(gpu) --mode test --init-from $(from) --output-dir eval/$(exp) --test-split $(test-split) --dropout 0.0 --cheat $(cheat_rate) --word-dropout 0 --use-last

test:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size $(bs) --gpu-id $(gpu) --mode test --init-from $(from) --output-dir eval/$(exp) --test-split $(test-split) --dropout 0.0 --cheat $(cheat_rate) --word-dropout 0 --max-num-examples $(num_ex)

summarize:
	python scripts/summarize_results.py --runs-dir $(exp)
