from .dataset import MRPCDataset, QQPDataset, RTEDataset, \
    STSBDataset, \
    QNLIDataset, COLADataset, SNLIDataset, MNLIDataset, WNLIDataset, SSTDataset, \
    SNLIHaohanDataset, MNLIStressTestDataset, MNLIHansDataset

tasks = {
    'MRPC': MRPCDataset,
    'QQP': QQPDataset,
    'QNLI': QNLIDataset,
    'RTE': RTEDataset,
    'STS-B': STSBDataset,
    'CoLA': COLADataset,
    'MNLI': MNLIDataset,
    'MNLI-stress': MNLIStressTestDataset,
    'MNLI-hans': MNLIHansDataset,
    'SNLI': SNLIDataset,
    'SNLI-haohan': SNLIHaohanDataset,
    'WNLI': WNLIDataset,
    'SST': SSTDataset
}

