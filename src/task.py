from .dataset import SNLIDataset, MNLIDataset, \
    SNLIHaohanDataset, MNLIStressTestDataset, MNLIHansDataset, \
    SNLIBreakDataset, SNLISwapDataset, MNLISwapDataset, SICKDataset, \
    QQPWangDataset, QQPPawsDataset, WikiPawsDataset

tasks = {
    'MNLI': MNLIDataset,
    'MNLI-stress': MNLIStressTestDataset,
    'MNLI-hans': MNLIHansDataset,
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

