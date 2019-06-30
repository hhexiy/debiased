**Train NLI models:

- BERT: `make train-bert exp=snli/bert task=SNLI test-split=dev`
- ESIM: `make train-esim exp=snli/esim task=SNLI test-split=dev lr=1e-4 nepochs=30 drop=0.5`
- Decomposable Attention: `make train-da exp=snli/da task=SNLI test-split=dev lr=1e-4 nepochs=30` 

**Test NLI models on challenge dataset:

- SNLI: `make test test-split=test from=[path to model] task=SNLI`
- HANS: `make test test-split=test from=[path to model] task=SNLI-hans`
