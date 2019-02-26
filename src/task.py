from .dataset import MRPCDataset, QQPDataset, RTEDataset, \
    STSBDataset, \
    QNLIDataset, COLADataset, SNLIDataset, MNLIDataset, WNLIDataset, SSTDataset, \
    SNLIHaohanDataset

tasks = {
    'MRPC': MRPCDataset,
    'QQP': QQPDataset,
    'QNLI': QNLIDataset,
    'RTE': RTEDataset,
    'STS-B': STSBDataset,
    'CoLA': COLADataset,
    'MNLI': MNLIDataset,
    'SNLI': SNLIDataset,
    'SNLI-haohan': SNLIHaohanDataset,
    'WNLI': WNLIDataset,
    'SST': SSTDataset
}

