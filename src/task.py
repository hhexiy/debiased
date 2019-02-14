from .dataset import MRPCDataset, QQPDataset, RTEDataset, \
    STSBDataset, \
    QNLIDataset, COLADataset, SNLIDataset, MNLIDataset, WNLIDataset, SSTDataset

tasks = {
    'MRPC': MRPCDataset,
    'QQP': QQPDataset,
    'QNLI': QNLIDataset,
    'RTE': RTEDataset,
    'STS-B': STSBDataset,
    'CoLA': COLADataset,
    'MNLI': MNLIDataset,
    'SNLI': SNLIDataset,
    'WNLI': WNLIDataset,
    'SST': SSTDataset
}

