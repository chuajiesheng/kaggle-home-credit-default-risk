import warnings
from multiprocessing import cpu_count

from joblib import Parallel
from bayes_opt import BayesianOptimization
import lightgbm as lgb

import feature as f

warnings.filterwarnings('ignore')


ID_COLUMN = 'SK_ID_CURR'
LABEL_COLUMN = 'TARGET'

n_threads = n_jobs = round(cpu_count() * 2 * 0.75)
n_jobs = cpu_count()
verbose = 1

X, y, X_test, train_test, bureau, bureau_bal, prev, credit_card_bal, pos_cash, installment_payment = f.read_dataset()
feature_mapping = f.get_feature_mapping(train_test, bureau, bureau_bal, prev, credit_card_bal, pos_cash, installment_payment)
features = Parallel(n_jobs=n_jobs, verbose=verbose)(feature_mapping)

for df in features:
    X = X.merge(right=df, how='left', on=ID_COLUMN)
    X_test = X_test.merge(right=df, how='left', on=ID_COLUMN)
    assert X.shape[0] == 307511

print('X.shape', X.shape)
print('X_test.shape', X_test.shape)

#
# Delete customer Id
#

del X['SK_ID_CURR']
del X_test['SK_ID_CURR']

#
# Prepare data
#

data = lgb.Dataset(X, label=y)

num_rounds = 3000
random_state = 42
num_iter = 1000
init_points = 5
params = {
    'learning_rate': 0.1,
    'verbose_eval': True,
    'seed': random_state,
    'nthread': n_threads,
    'boosting': 'gbdt',
    'metric': 'auc',
    'seed': 42,
    'is_unbalance': True,
    'device': 'cpu'
}


def lgb_evaluate(max_bin, min_child_samples, learning_rate, num_leaves, min_child_weight, colsample_bytree, max_depth, subsample, reg_lambda, reg_alpha, min_split_gain):

    params['max_bin'] = int(max(max_bin, 1))
    params['min_child_samples'] = int(max(min_child_samples, 0))
    params['learning_rate'] = max(learning_rate, 0)
    params['num_leaves'] = int(num_leaves)
    params['min_child_weight'] = int(min_child_weight)
    params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['reg_lambda'] = max(reg_lambda, 0)
    params['reg_alpha'] = max(reg_alpha, 0)
    params['min_split_gain'] = max(min_split_gain, 0)

    cv_result = lgb.cv(params,
                       data,
                       nfold=5,
                       num_boost_round=num_rounds,
                       early_stopping_rounds=50,
                       verbose_eval=100,
                       seed=random_state,
                       show_stdv=True)

    return max(cv_result['auc-mean'])


lgbBO = BayesianOptimization(lgb_evaluate, {'max_bin': (2**3, 2**11),
                                            'min_child_samples': (10, 50),
                                            'learning_rate': (0.001, 0.5),
                                            'num_leaves': (2, 2 ** 10),
                                            'min_child_weight': (1, 22),
                                            'colsample_bytree': (0.2, 1),
                                            'max_depth': (5, 15),
                                            'subsample': (0.2, 1),
                                            'reg_lambda': (0, 100),
                                            'reg_alpha': (0, 100),
                                            'min_split_gain': (0.01, 0.1),
                                            })

lgbBO.maximize(init_points=init_points, n_iter=num_iter, acq="poi", xi=0.1)
