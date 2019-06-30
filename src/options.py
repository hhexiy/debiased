def add_default_arguments(parser):
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU id (-1 means CPU)')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='train or test')

def add_data_arguments(parser):
    group = parser.add_argument_group('Data')
    #group.add_argument('--train-file', default='snli_1.0/snli_1.0_train.txt',
    #                    help='training set file')
    #group.add_argument('--test-file', default='snli_1.0/snli_1.0_dev.txt',
    #                    help='validation set file')
    group.add_argument('--train-split', default='train',
                        help='training data split name')
    group.add_argument('--test-split', default='dev',
                        help='test data split name')
    group.add_argument('--superficial', choices=['hypothesis', 'handcrafted'], default=None,
                        help='only use superficial features')
    group.add_argument('--additive', nargs='*', default=None,
                        help='path to models to be added to the additive model')
    group.add_argument('--project', action='store_true',
                        help='project out previous models')
    group.add_argument('--remove', action='store_true',
                        help='remove examples that are predicted correctly by previous models')
    group.add_argument('--remove-cheat', choices=['True', 'False'], default='False',
                        help='remove examples where cheating features are enabled')
    group.add_argument('--cheat', type=float, default=-1,
                        help='percentage of training data where value of the cheating feature is the groundtruth label. -1 means no cheating features is added at all.')
    group.add_argument('--task-name', required=True,
                        help='The name of the task to fine-tune.(MRPC,...)')
    group.add_argument('--max-num-examples', type=int, default=-1,
                        help='maximum number of examples to read, -1 means all.')

def add_logging_arguments(parser):
    group = parser.add_argument_group('Logging')
    group.add_argument('--exp-id', default=None,
                        help='experiment ID')
    group.add_argument('--output-dir', default='.',
                        help='output directory')
    group.add_argument('--log-interval', type=int, default=20,
                        help='the interval of two print')

def add_model_arguments(parser):
    group = parser.add_argument_group('Model')
    #group.add_argument('--num-classes', type=int, default=2,
    #                    help='number of classes')
    group.add_argument('--model-type', choices=['cbow', 'bert', 'da', 'esim'], default='bert',
                        help='core classifier type')
    #group.add_argument('--embedding', default='glove',
    #                    help='word embedding type')
    group.add_argument('--embedding-source', default='glove.840B.300d',
                        help='embedding file source')
    group.add_argument('--embedding-size', type=int, default=300,
                        help='size of pretrained word embedding')
    group.add_argument('--hidden-size', type=int, default=200,
                        help='hidden layer size')
    group.add_argument('--num-layers', type=int, default=1,
                        help='number of hidden layers')
    #group.add_argument('--intra-attention', action='store_true',
    #                    help='use intra-sentence attention')
    group.add_argument('--max-len', type=int, default=128,
                        help='Maximum length of the sentence pairs')
    group.add_argument('--init-from',
                        help='directory to load model')
    group.add_argument('--use-last', action='store_true',
                        help='use the last model instead of the best modal on the dev set')
    group.add_argument('--additive-mode', choices=['prev', 'last', 'all'], default='all',
                        help='use which classifier')
    group.add_argument('--word-dropout', type=float, default=0,
                        help='word dropout rate.')
    group.add_argument('--word-dropout-region', nargs='+', default=None,
                        help='where to dropout words. None means everywhere.')

def add_training_arguments(parser):
    group = parser.add_argument_group('Training')
    group.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    group.add_argument('--eval-batch-size', type=int, default=128,
                        help='inference time batch size')
    group.add_argument('--lr', type=float, default=5e-5,
                        help='learning rate')
    group.add_argument('--epochs', type=int, default=10,
                        help='maximum number of epochs to train')
    group.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    group.add_argument('--weight-decay', type=float, default=0.,
                        help='l2 regularization weight')
    group.add_argument('--fix-word-embedding', action='store_true',
                        help='fix pretrained word embedding during training')
    group.add_argument('--optimizer', default='adam',
                        help='optimization algorithm')
    group.add_argument('--warmup-ratio', type=float, default=0.1,
                        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    group.add_argument('--noising-by-epoch', action='store_true',
                        help='should data noising (e.g. word dropout) be applied every epoch')
    group.add_argument('--early-stop', action='store_true',
                        help='can stop early before max number of epochs is reached')

def check_arguments(args):
    if args.superficial == 'handcrafted':
        assert args.model_type == 'cbow'
