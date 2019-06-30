from .dataset import SNLIDataset, MNLIDataset, \
    SNLIHaohanDataset, MNLIStressTestDataset, MNLIHansDataset, \
    SNLIBreakDataset, SNLISwapDataset, MNLISwapDataset

tasks = {
    'MNLI': MNLIDataset,
    'MNLI-stress': MNLIStressTestDataset,
    'MNLI-hans': MNLIHansDataset,
    'SNLI': SNLIDataset,
    'SNLI-haohan': SNLIHaohanDataset,
    'SNLI-break': SNLIBreakDataset,
    'SNLI-swap': SNLISwapDataset,
    'MNLI-swap': MNLISwapDataset,
}

