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
handlers = [logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()]
logging.basicConfig(level=level, handlers=handlers, format='%(message)s')

logging.info("Started. Results will be in {}".format(save_path))
logging.info(args)

with open(f'{folder}/train.pkl', 'rb') as f:
    data_train = pickle.load(f)

seed = 0
random.seed(seed)
torch.manual_seed(seed)
args.device = None
if args.cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(seed)
else:
    args.device = torch.device('cpu')

if args.noncontext:
    dsloader = NonContextDataLoader(data_train, None, label_size=1, target=args.target)
else:
    dsloader = ContextDataLoader(data_train, None, label_size=args.label_size, target=args.target)

train_set = dsloader.train_set
val_set = dsloader.val_set
mm = ModelManager(train_set, val_set, None, save_path, args)
Model = NSMMPP
hidden_choices = args.nhid
tuning_file = os.path.join(save_path, 'tuning.pkl')
if os.path.exists(tuning_file):
    with open(tuning_file, 'rb') as f:
        tuning_results = pickle.load(f)
else:
    tuning_results = []
    for h in hidden_choices:
        logging.info("training for hidden size {}".format(h))
        model = Model(dsloader.label_size, h, args)
        loss, epochs = mm.train(model, name='model_hidden_{}.pt'.format(h))
        logging.info('model |hidden| = {}, epochs = {}, loss = {}'.format(h, epochs, loss))
        tuning_results.append({
            'hidden': h,
            'epochs': epochs,
            'loss': loss,
        })
    with open(tuning_file, 'wb') as f:
        pickle.dump(tuning_results, f)
best_result = tuning_results[0]
for item in tuning_results:
    if item['loss'] < best_result['loss']:
        best_result = item
logging.info('best result is {}'.format(best_result))
model = Model(dsloader.label_size, best_result['hidden'], args)
model_name = 'best_model.pt'
try:
    mm.load_model(model, model_name)
except Exception as e:
    logging.info(str(e))
    loss, epochs = mm.train(model, epochs=best_result['epochs'], use_all_data=True, name=model_name)
best_model = model
logging.info("Finished. Results are in {}".format(save_path))
