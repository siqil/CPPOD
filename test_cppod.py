from cppod import *
import argparse
import datetime

parser = argparse.ArgumentParser()

parser.add_argument('--nhid', type=int, nargs='+',
                    default=[64, 128, 256, 512, 1024],
                    help='number of hidden units per layer')
parser.add_argument('--lr', type=float,
                    default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int,
                    default=500,
                    help='max number of epochs')
parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='batch size')
parser.add_argument('--sample_multiplier', type=int, default=10, metavar='N',
                    help='sample size multiplier')
parser.add_argument('--seed', type=int,
                    default=0,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='enable CUDA')
parser.add_argument('--log-interval', type=int,
                    default=10,
                    help='report interval')
parser.add_argument('--output', type=str,
                    default=None,
                    help='path to save the final model')
parser.add_argument('--target', type=int,
                    default=1,
                    help='the target event type')
parser.add_argument('--label-size', type=int, required=True,
                    help='the target event type')
parser.add_argument('--dataset', type=str, required=True,
                    help='the (folder) name of the dataset')
parser.add_argument('--ignore-first', type=bool,
                    default=False,
                    help='ignore first event in prediction')
parser.add_argument('--noncontext', action='store_true',
                    help='ignore contextual events')
parser.add_argument('--debug', action='store_true',
                    help='debug')

args = parser.parse_args()

data_set = args.dataset

folder = f'data/{data_set}'
if args.noncontext:
    save_path = f'model/{data_set}/NH'
else:
    save_path = f'model/{data_set}/CNH'
log_path = os.path.join(save_path, 'log')
level = logging.DEBUG
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(log_path, exist_ok=True)
log_file_path = os.path.join(log_path, '{}.log'.format(timestamp))
handlers = [#logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()]
logging.basicConfig(level=level, handlers=handlers, format='%(message)s')
logging.info(args)
seed = 0
random.seed(seed)
torch.manual_seed(seed)
args.device = None
if args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
else:
    args.device = torch.device('cpu')

debug = args.debug
outlier_types = ['commiss', 'omiss']
ps = ["0.1", "0.05", "sin", "pc"]

with open(f'{folder}/train.pkl', 'rb') as f:
    data_train = pickle.load(f)

for outlier in outlier_types:
    instant = (outlier == 'commiss')
    for p in ps:
        with open(f'{folder}/test_{outlier}_{p}.pkl', 'rb') as f:
            data_test = pickle.load(f)
        if args.noncontext:
            dsloader = NonContextDataLoader(data_train, data_test, label_size=1, target=args.target)
        else:
            dsloader = ContextDataLoader(data_train, data_test, label_size=args.label_size, target=args.target)
        train_set = dsloader.train_set
        val_set = dsloader.val_set
        test_set = dsloader.test_set
        mm = ModelManager(train_set, val_set, test_set, save_path, args)
        Model = NSMMPP
        hidden_choices = args.nhid
        tuning_file = os.path.join(save_path, 'tuning.pkl')
        with open(tuning_file, 'rb') as f:
            tuning_results = pickle.load(f)
        best_result = tuning_results[0]
        for item in tuning_results:
            if item['loss'] < best_result['loss']:
                best_result = item
        logging.info('best result is {}'.format(best_result))
        model = Model(dsloader.label_size, best_result['hidden'], args)
        model_name = 'best_model.pt'
        mm.load_model(model, model_name)
        model.debug = False
        result = mm.detect_outlier(model, instant=instant, debug=debug)
        result_path = f'result/{data_set}/{outlier}'
        if not debug:
            if args.noncontext:
                result.to_csv(f'{result_path}/NH_{p}.csv')
            else:
                result.to_csv(f'{result_path}/CNH_{p}.csv')
