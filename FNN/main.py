from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys

if sys.version[0] == '2':
    import cPickle as pkl
else:
    import pickle as pkl

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import os

p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not p in sys.path:
    sys.path.append(p)

import utils
from FNN import FM, FNN

# In[1]: data setting
# ffm = pd.read_csv('./intern_data/train_ffm.csv')
# train_file,test_file = train_test_split(ffm,test_size = 0.2)
train_file = 'data/train_ffm.csv'
test_file = 'data/test_ffm.csv'
pkl_path = 'model/fm.pkl'
# test_final_file = './data/test_ffm_final.csv'
input_dim = utils.INPUT_DIM

train_data = utils.read_data(train_file)
train_data = utils.shuffle(train_data)
test_data = utils.read_data(test_file)

if train_data[1].ndim > 1:
    print('label must be 1-dim')
    exit(0)
print('read finish')
print('train data size:', train_data[0].shape)
print('test data size:', test_data[0].shape)

train_size = train_data[0].shape[0]
test_size = test_data[0].shape[0]
num_feas = len(utils.FIELD_SIZES)

min_round = 1
num_round = 10
early_stop_round = 10
batch_size = 1024
iters = int((train_size + batch_size - 1) / batch_size)

field_sizes = utils.FIELD_SIZES
field_offsets = utils.FIELD_OFFSETS

print("field_size", field_sizes)
print("field_offsets", field_offsets)
# In[2]: model configuration
# algo = 'fm'
algo = 'fnn'
if algo in {'fnn', 'anfm', 'amlp', 'ccpm', 'pnn1', 'pnn2'}:
    train_data = utils.split_data(train_data)
    test_data = utils.split_data(test_data)
    tmp = []
    for x in field_sizes:
        if x > 0:
            tmp.append(x)
    field_sizes = tmp
    print('remove empty fields', field_sizes)

if algo == 'fm':
    fm_params = {
        'input_dim': input_dim,
        'factor_order': 128,
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'l2_w': 0,
        'l2_v': 0,
    }
    print(fm_params)
    model = FM(**fm_params)
elif algo == 'fnn':
    fnn_params = {
        'field_sizes': field_sizes,
        'embed_size': 129,
        'layer_sizes': [500, 1],
        'layer_acts': ['relu', None],
        'drop_out': [0, 0],
        'opt_algo': 'gd',
        'learning_rate': 0.1,
        'embed_l2': 0,
        'layer_l2': [0, 0],
        'random_seed': 0,
        'init_path': pkl_path,
    }
    print(fnn_params)
    model = FNN(**fnn_params)


# In[3]: model training
def train(model):
    history_score = []
    for i in range(num_round):
        fetches = [model.optimizer, model.loss]
        # if model.model_file is not None:
        #	model.saver.restore(model.sess,model.model_file)
        if batch_size > 0:
            ls = []
            print('[%d]\ttraining...' % i)
            for j in range(iters):
                X_i, y_i = utils.slice(train_data, j * batch_size, batch_size)
                _, l = model.run(fetches, X_i, y_i)
                ls.append(l)
        #                if(j == 0):
        #                    print('---------------------------')
        #                    print(X_i[0])
        #                    print('---------------------------')

        elif batch_size == -1:
            X_i, y_i = utils.slice(train_data)
            print('xi.shape', X_i.shape)
            print('yi.shape', y_i.shape)
            _, l = model.run(fetches, X_i, y_i)
            ls = [l]
        # model.saver.save(model.sess,'./Save/train.ckpt')
        train_preds = []
        print('[%d]\tevaluating...' % i)
        for j in range(int(train_size / 10000 + 1)):
            X_i, _ = utils.slice(train_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            train_preds.extend(preds)
        test_preds = []
        for j in range(int(test_size / 10000 + 1)):
            X_i, _ = utils.slice(test_data, j * 10000, 10000)
            preds = model.run(model.y_prob, X_i, mode='test')
            test_preds.extend(preds)
        train_score = roc_auc_score(train_data[1], train_preds)
        test_score = roc_auc_score(test_data[1], test_preds)
        print('[%d]\tloss (with l2 norm):%f\ttrain-auc: %f\teval-auc: %f' % (i, np.mean(ls), train_score, test_score))
        history_score.append(test_score)
        if i > min_round and i > early_stop_round:
            if np.argmax(history_score) == i - early_stop_round and history_score[-1] - history_score[
                -1 * early_stop_round] < 1e-5:
                print('early stop\nbest iteration:\n[%d]\teval-auc: %f' % (
                    np.argmax(history_score), np.max(history_score)))
                break
    #    var_map = {}
    #    for name, var in model.vars.items():
    #        print(name, var)
    #        ans = model.run(var)
    #        var_map[name] = ans
    #        print(var_map[name])
    if algo == 'fm':
        model.dump(pkl_path)


print('iters', iters)
train(model)
