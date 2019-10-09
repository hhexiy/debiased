# Unlearn dataset bias by fitting the residual
This repository (the `drift` branch) contains code for the paper
> **Unlearn dataset bias for natural language inference by fitting the residual**.
> He He, Sheng Zha, and Haohan Wang.
> In proceedings of the DeepLo Workshop at EMNLP 2019.
> https://arxiv.org/abs/1908.10763

## Dependencies
- Python 3.6
- [MXNet 1.5.0](https://mxnet.apache.org/get_started/index.html?version=v1.5.0&platform=linux&language=python&environ=pip&processor=gpu), e.g., using cuda-10.0, `pip install mxnet-cu100`
- [GluonNLP 0.8.0](https://github.com/dmlc/gluon-nlp/)

Install all Python packages: `pip install -r requirements.txt`

## Data
- SNLI, MNLI
```
mkdir -p data/glue_data
python scripts/download_glue_data.py --tasks MNLI --data_dir data/glue_data
python scripts/download_glue_data.py --tasks SNLI --data_dir data/glue_data
```
- [HANS](https://github.com/tommccoy1/hans)
```
git clone https://github.com/tommccoy1/hans.git
mkdir -p data/glue_data/MNLI-hans
python scripts/hans_to_glue.py --hans-data hans/heuristics_evaluation_set.txt --outdir   data/glue_data/MNLI-hans
rm -rf hans
```

Datasets will be found in `data/glue_data/{SNLI,MNLI,MNLI-hans}`.

## Code
Entry point: `src/main.py`.

### Main options
Complete options are documented in `src/options.py`.

- `task`: which dataset to load.
- `test-split`: which split to use for validation (training) or evaluation (testing).
- `output-dir`: directory to save models and artifacts. A UUID will be automatically generated to create a subdirectory under `output-dir`.
- `model-type`: which model to use, e.g. `bert`, `cbow` etc.
- `max-num-examples`: maximum number of examples to load.

**NOTE**: The code will automatically download files (pretraind models, embeddings etc.) through MXNet. These files will be saved in `MXNET_HOME` (default directory is `~/.mxnet`), which can take up a large space. You might want to set `MXNET_HOME` to a different directory.

We use MNLI as the example training data below, but you can easily switch to SNLI by modifying the following options: `--task SNLI --test-split dev`.

We use a default batch size of 32 and the Adam optimizer.

### <a name="mle"></a>Training NLI models by MLE (baselines)
- BERT finetuning
```
MXNET_HOME=.mxnet MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data \
python -m src.main --task-name MNLI --test-split dev_matched \
--epochs 4 --lr 2e-5 --log-interval 5000 --output-dir output/MNLI \
--model-type bert
```
- Decomposable Attention
```
MXNET_HOME=.mxnet MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data \
python -m src.main --task-name MNLI --test-split dev_matched \
--epochs 30 --lr 1e-4 --log-interval 5000 --output-dir output/MNLI \
--model-type da --hidden-size 300 --early-stop
```
- ESIM
```
MXNET_HOME=.mxnet MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data \
python -m src.main --task-name MNLI --test-split dev_matched \
--epochs 30 --lr 1e-4 --log-interval 5000 --output-dir output/MNLI \
--dropout 0.5 --model-type esim --hidden-size 300 --early-stop --max-len 64
```

### <a name="biased"></a>Training biased models
- Hypothesis-only (finetuned BERT with the hypothesis sentence as the input)
```
MXNET_HOME=.mxnet MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data \
python -m src.main --task-name MNLI --test-split dev_matched \
--epochs 4 --lr 2e-5 --log-interval 5000 --output-dir output/MNLI \
--model-type bert --superficial hypothesis
```
- CBOW
```
MXNET_HOME=.mxnet MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data \
python -m src.main --task-name MNLI --test-split dev_matched \
--epochs 40 --lr 1e-4 --log-interval 5000 --output-dir output/MNLI \
--model-type cbow --early-stop
```
- Handcrafted features
```
MXNET_HOME=.mxnet MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data \
python -m src.main --task-name MNLI --test-split dev_matched \
--epochs 40 --lr 1e-4 --log-interval 5000 --output-dir output/MNLI \
--model-type cbow --early-stop --superficial handcrafted
```

### Training NLI models by DRiFt (based on the biased models)
To train (debiased) models by the residual fitting algorithm, DRiFt,
we need to have the [biased models](#biased) ready. Then, we reuse the [MLE training script](#mle) for BERT, DA, and ESIM, with the following updated/new options:
- `--additive-mode all`: learning with an ensemble of the (fixed) biased model and the debiased model.
- `--additive <path>`: path to directory of the biased model.
- `--epochs <num_epochs>`: 8 for BERT; 40 for DA and ESIM.

### Training NLI models by RM (based on the biased models)
RM simply removes examples predicted correctly by DRiFt. Similarly, we reuse the [MLE training script](#mle) for BERT, DA, and ESIM, with the following updated/new options:
- `--additive-mode last`: equivalent to learning the debiased model by MLE.
- `--additive <path>`: path to directory of the biased model.
- `--epochs <num_epochs>`: 8 for BERT; 40 for DA and ESIM.
- `--remove`: remove biased examples.

### Training models on data with synthesized cheating features
- `--cheat <cheat_rate>`: during training, set the cheating rate. At test time, set cheating rate to `0` such that examples will be prepended with random lables. A cheating rate of `-1` means that no cheating feature is added, i.e. normal data.
- `--remove-cheat`: remove cheated examples during training.

### Evaluation on HANS
- `--test-split <HANS-split>`: for HANS, the valid splits are `lexical_overlap`, `subsequence`, `constituent`.
- `--additive-mode last`: only used the debiased model for testing. For models trained by MLE, this option has no effect.
- `--output-dir <path>`: logs and predictions for each example (`predictions.tsv`) will be saved in `path`.
```
MXNET_HOME=.mxnet MXNET_GPU_MEM_POOL_TYPE=Round GLUE_DIR=data/glue_data \
python -m src.main --task-name HANS --eval-batch-size 128 --mode test \
--init-from <path> --output-dir eval/HANS --test-split <HANS-split> \
--additive-mode last
```

### Utilities
- `scripts/summarize_results.py`: summarize evaluation results from multiple runs and print in a tabular form.
- `scripts/error_analysis.py`: simple classification error statistics based on `predictions.tsv`.
