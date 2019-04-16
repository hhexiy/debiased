exp=debug
model=cbow
gpu=0
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
nepochs=3

preprocess:
	for split in train dev test; do python3 scripts/preprocess.py --input data/snli_1.0/snli_1.0_$$split.txt --output data/snli_1.0/$$split.txt; done

train-attn:
	python3 -m src.main --train-file data/snli_1.0/train.txt --test-file data/snli_1.0/dev.txt --exp-id $(exp) --output-dir output --batch-size 32 --print-interval 5000 --lr 0.025 --epochs 300 --gpu-id $(gpu) --dropout 0.2 --weight-decay 1e-5 --model-type dec-attn --fix-word-embedding #--superficial #--max-num-examples 1000

train-bert:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python3 -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --noising-by-epoch --model-type $(model) 

train-cbow:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs $(nepochs) --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) --word-dropout $(wdrop) --noising-by-epoch --model-type cbow

train-bert-sup:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python3 -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --superficial --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) 

train-bert-bow:
	MXNET_HOME=$(mxnet_home) MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python3 -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --superficial --test-split $(test-split) --cheat $(cheat_rate) --max-num-examples $(num_ex) 

train-bert-additive:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python3 -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --additive $(from) --cheat $(cheat_rate) --max-num-examples $(num_ex)

train-bert-noise:
	MXNET_GPU_MEM_POOL_TYPE=Round MXNET_HOME=$(mxnet_home) GLUE_DIR=data/glue_data python3 -m src.main --task-name $(task) --batch-size $(bs) --optimizer bertadam --epochs 4 --gpu-id $(gpu) --lr $(lr) --log-interval $(interval) --output-dir output/$(exp) --dropout 0.1 --test-split $(test-split) --max-num-examples $(num_ex) --noising-by-epoch --word-dropout $(wdrop) --cheat $(cheat_rate)

test-bert:
	MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data python -m src.main --task-name $(task) --eval-batch-size $(bs) --gpu-id $(gpu) --mode test --init-from $(from) --output-dir output/$(exp) --test-split $(test-split) --use-last --dropout 0.0 --cheat $(cheat_rate)

summarize:
	python scripts/summarize_results.py --runs-dir $(exp)
