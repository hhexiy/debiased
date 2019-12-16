from .dataset import SNLIDataset, MNLIDataset, \
    SNLIHaohanDataset, MNLIStressTestDataset, MNLIHansDataset, MNLILenDataset, \
    SNLIBreakDataset, SNLISwapDataset, MNLISwapDataset, SICKDataset, \
    QQPWangDataset, QQPPawsDataset, WikiPawsDataset, MNLINoSubsetDataset

tasks = {
    'MNLI': MNLIDataset,
    'MNLI-stress': MNLIStressTestDataset,
    'MNLI-hans': MNLIHansDataset,
    'MNLI-length': MNLILenDataset,
    'MNLI-no-subset': MNLINoSubsetDataset,
    'SNLI': SNLIDataset,
    'SNLI-haohan': SNLIHaohanDataset,
    'SNLI-break': SNLIBreakDataset,
    'SNLI-swap': SNLISwapDataset,
    'MNLI-swap': MNLISwapDataset,
    'QQP-wang': QQPWangDataset,
    'QQP-paws': QQPPawsDataset,
    'Wiki-paws': WikiPawsDataset,
    'SICK': SICKDataset,
}

