import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from scipy.stats import norm, randint
import lightgbm as lgb
import xgboost as xgb
import os
import gc
from multiprocessing import cpu_count
from datetime import datetime

import utility.random_grid as random_grid
import utility.expand_grid as expand_grid
import fit_models as fit_models
import predict_models as predict_models

DATA_DIR = '{}/data'.format(os.getcwd())
SUBMISSION_DIR = '{}/submission'.format(os.getcwd())
os.listdir(DATA_DIR)

ID_COLUMN = 'SK_ID_CURR'
LABEL_COLUMN = 'TARGET'
N_THREADS = round(cpu_count() * 2 * 0.8)

INPUT_FILE = os.path.join(DATA_DIR, 'application_train.csv.zip')
X = pd.read_csv(INPUT_FILE)
y = X[LABEL_COLUMN]
del X['TARGET']

TEST_INPUT_FILE = os.path.join(DATA_DIR, 'application_test.csv.zip')
X_test = pd.read_csv(TEST_INPUT_FILE)

BUREAU_FILE = os.path.join(DATA_DIR, 'bureau.csv.zip')
BUREAU_BAL_FILE = os.path.join(DATA_DIR, 'bureau_balance.csv.zip')
PREV_APPLICATION_FILE = os.path.join(DATA_DIR, 'previous_application.csv.zip')
CREDIT_CARD_BAL_FILE = os.path.join(DATA_DIR, 'credit_card_balance.csv.zip')
POS_CASH_FILE = os.path.join(DATA_DIR, 'POS_CASH_balance.csv.zip')
INSTALLMENT_PAYMENT_FILE = os.path.join(DATA_DIR, 'installments_payments.csv.zip')
SUBMISSION_FILE = os.path.join(DATA_DIR, 'sample_submission.csv.zip')

bureau = pd.read_csv(BUREAU_FILE)
bureau_bal = pd.read_csv(BUREAU_BAL_FILE)
prev = pd.read_csv(PREV_APPLICATION_FILE)
credit_card_bal = pd.read_csv(CREDIT_CARD_BAL_FILE)
pos_cash = pd.read_csv(POS_CASH_FILE)
installment_payment = pd.read_csv(INSTALLMENT_PAYMENT_FILE)

#
# Feature Generation
#

categorical_features = [col for col in X.columns if X[col].dtype == 'object']

one_hot_df = pd.concat([X, X_test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_features)

X = one_hot_df.iloc[:X.shape[0], :]
X_test = one_hot_df.iloc[X.shape[0]:, ]

#
# Generated features from previous application
#

no_prev_app = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count()
no_prev_app.describe()

sorted_prev = installment_payment[['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']].sort_values(
    ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
compare_to_last = sorted_prev.groupby(by=['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].diff().fillna(1)
sorted_prev['COMPARED_TO_LAST'] = compare_to_last
std_of_installment_seq = sorted_prev[['SK_ID_PREV', 'COMPARED_TO_LAST']].groupby(by=['SK_ID_PREV']).std().reset_index()
std_of_installment_seq = std_of_installment_seq.rename(index=str,
                                                       columns={'COMPARED_TO_LAST': 'STD_OF_INSTALLMENT_SEQ'})

prev_installment_feature = installment_payment[['SK_ID_CURR', 'SK_ID_PREV']].copy()
prev_installment_feature = prev_installment_feature.merge(right=std_of_installment_seq, how='left', on='SK_ID_PREV')

late_installment = installment_payment[
    ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']].sort_values(['SK_ID_CURR', 'SK_ID_PREV'])
late_installment['LATE'] = late_installment['DAYS_INSTALMENT'] - late_installment['DAYS_ENTRY_PAYMENT']

late_mean = late_installment[['SK_ID_PREV', 'LATE']].groupby(by=['SK_ID_PREV']).mean().fillna(0).reset_index()
late_mean = late_mean.rename(index=str, columns={'LATE': 'MEAN_OF_LATE_INSTALLMENT'})

prev_installment_feature = prev_installment_feature.merge(right=late_mean, how='left', on='SK_ID_PREV')

pay_less = installment_payment[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT']].sort_values(
    ['SK_ID_CURR', 'SK_ID_PREV'])
pay_less['INSUFFICIENT_PAYMENT'] = pay_less['AMT_INSTALMENT'] - pay_less['AMT_PAYMENT']
pay_less = pay_less[['SK_ID_PREV', 'INSUFFICIENT_PAYMENT']].groupby(by=['SK_ID_PREV']).mean().fillna(0).reset_index()
pay_less = pay_less.rename(index=str, columns={'INSUFFICIENT_PAYMENT': 'MEAN_OF_INSUFFICIENT_PAYMENT'})

prev_installment_feature = prev_installment_feature.merge(right=pay_less, how='left', on='SK_ID_PREV')

prev_installment_feature_by_curr = prev_installment_feature.groupby(by=['SK_ID_CURR']).mean().fillna(0).reset_index()

del prev_installment_feature_by_curr['SK_ID_PREV']

X = X.merge(right=prev_installment_feature_by_curr, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=prev_installment_feature_by_curr, how='left', on='SK_ID_CURR')

print('prev_installment_feature', X.shape)
assert X.shape[0] == 307511

del no_prev_app
del sorted_prev
del compare_to_last
del std_of_installment_seq
del prev_installment_feature
del late_installment
del late_mean
del prev_installment_feature
del prev_installment_feature_by_curr
gc.collect()

#
# - bureau.csv
#     - All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
#     - For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
#
# - bureau_balance.csv
#     - Monthly balances of previous credits in Credit Bureau.
#     - This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.
#

#
# count_day_credit
#

count_day_credit = bureau[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].count()
count_day_credit = count_day_credit.reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})

X = X.merge(right=count_day_credit, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=count_day_credit, how='left', on='SK_ID_CURR')

print('count_day_credit', X.shape)
del count_day_credit
gc.collect()

#
# count_credit_type
#

count_credit_type = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique()
count_credit_type = count_credit_type.reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})

X = X.merge(right=count_credit_type, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=count_credit_type, how='left', on='SK_ID_CURR')

print('count_credit_type', X.shape)
del count_credit_type
gc.collect()

#
# credit_variety
#

credit_variety = count_day_credit.merge(right=count_credit_type, how='left', on='SK_ID_CURR')
credit_variety['AVERAGE_LOAN_TYPE'] = credit_variety['BUREAU_LOAN_COUNT'] / credit_variety['BUREAU_LOAN_TYPES']
del credit_variety['BUREAU_LOAN_COUNT'], credit_variety['BUREAU_LOAN_TYPES']

X = X.merge(right=credit_variety, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=credit_variety, how='left', on='SK_ID_CURR')

print('credit_variety', X.shape)
del count_day_credit
del count_credit_type
del credit_variety


#
# bureau_active_sum
#


def count_active(x):
    if x == 'Closed':
        y = 0
    else:
        y = 1
    return y


active = bureau[['SK_ID_CURR', 'CREDIT_ACTIVE']].apply(lambda x: count_active(x.CREDIT_ACTIVE), axis=1)

bureau_active = bureau[['SK_ID_CURR']].copy()
bureau_active['ACTIVE_COUNT'] = active

bureau_active_sum = bureau_active.groupby(by=['SK_ID_CURR'])['ACTIVE_COUNT'].sum()

X = X.merge(right=bureau_active_sum.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=bureau_active_sum.reset_index(), how='left', on='SK_ID_CURR')

print('bureau_active_sum', X.shape)
del active
del bureau_active_sum
gc.collect()

assert X.shape[0] == 307511

#
# bureau_active_mean
#

bureau_active_mean = bureau_active.groupby(by=['SK_ID_CURR'])['ACTIVE_COUNT'].mean().reset_index()
bureau_active_mean = bureau_active_mean.rename(index=str, columns={'ACTIVE_COUNT': 'ACTIVE_LOANS_PERCENTAGE'})

X = X.merge(right=bureau_active_mean, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=bureau_active_mean, how='left', on='SK_ID_CURR')

print('bureau_active_mean', X.shape)
del bureau_active
del bureau_active_mean

#
# day_credit_group
#

day_credit_group = bureau[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])
day_credit_group = day_credit_group.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(
    drop=True)

day_credit_group['DAYS_CREDIT1'] = day_credit_group['DAYS_CREDIT'] * -1
day_credit_group['DAYS_DIFF'] = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
day_credit_group['DAYS_DIFF'] = day_credit_group['DAYS_DIFF'].fillna(0).astype('uint32')

# Differ from shared
# https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering?scriptVersionId=3798236

del day_credit_group['DAYS_CREDIT1'], day_credit_group['DAYS_CREDIT'], day_credit_group['SK_ID_BUREAU']
day_credit_group_mean = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_DIFF'].mean()
day_credit_group_mean = day_credit_group_mean.reset_index().rename(index=str, columns={'DAYS_DIFF': 'MEAN_DAYS_DIFF'})

day_credit_group_max = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_DIFF'].max()
day_credit_group_max = day_credit_group_max.reset_index().rename(index=str, columns={'DAYS_DIFF': 'MAX_DAYS_DIFF'})

X = X.merge(right=day_credit_group_mean, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=day_credit_group_mean, how='left', on='SK_ID_CURR')

X = X.merge(right=day_credit_group_max, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=day_credit_group_max, how='left', on='SK_ID_CURR')

print('day_credit_group', X.shape)
del day_credit_group_mean
del day_credit_group_max
gc.collect()

assert X.shape[0] == 307511


#
# bureau_credit_time
#


def check_credit_time(x):
    if x < 0:
        y = 0
    else:
        y = 1
    return y


credit_time = bureau[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE']].apply(lambda x: check_credit_time(x.DAYS_CREDIT_ENDDATE),
                                                                  axis=1)
bureau_credit_time = bureau[['SK_ID_CURR']].copy()
bureau_credit_time['CREDIT_TIME'] = credit_time

credit_time_mean = bureau_credit_time.groupby(by=['SK_ID_CURR'])['CREDIT_TIME'].mean()
credit_time_mean = credit_time_mean.reset_index().rename(index=str, columns={'CREDIT_TIME': 'MEAN_CREDIT_TIME'})

credit_time_max = bureau_credit_time.groupby(by=['SK_ID_CURR'])['CREDIT_TIME'].max()
credit_time_max = credit_time_max.reset_index().rename(index=str, columns={'CREDIT_TIME': 'MAX_CREDIT_TIME'})

X = X.merge(right=credit_time_mean, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=credit_time_mean, how='left', on='SK_ID_CURR')

X = X.merge(right=credit_time_max, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=credit_time_max, how='left', on='SK_ID_CURR')

print('bureau_credit_time', X.shape)
del bureau_credit_time
del credit_time
del credit_time_mean
del credit_time_max
gc.collect()

assert X.shape[0] == 307511

#
# max_per_user
#
positive_credit_end_date = bureau[bureau['DAYS_CREDIT_ENDDATE'] > 0]
max_per_loan = positive_credit_end_date.groupby(by=['SK_ID_CURR', 'SK_ID_BUREAU'])[
    'DAYS_CREDIT_ENDDATE'].max().reset_index()
max_per_loan = max_per_loan.rename(index=str, columns={'DAYS_CREDIT_ENDDATE': 'MAX_DAYS_CREDIT_ENDDATE'})

max_per_user = max_per_loan.groupby(by=['SK_ID_CURR'])['MAX_DAYS_CREDIT_ENDDATE'].max().reset_index()

X = X.merge(right=max_per_user, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=max_per_user, how='left', on='SK_ID_CURR')

print('max_per_user', X.shape)
del max_per_user

assert X.shape[0] == 307511

#
# current_loan_count
#

current_loan_count = positive_credit_end_date.groupby(by=['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
current_loan_count = current_loan_count.rename(index=str, columns={'SK_ID_BUREAU': 'COUNT_SK_ID_BUREAU'})

X = X.merge(right=current_loan_count, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=current_loan_count, how='left', on='SK_ID_CURR')

print('current_loan_count', X.shape)
del current_loan_count
del positive_credit_end_date
gc.collect()

assert X.shape[0] == 307511

#
# cust_debt_to_credit
#

bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)

cust_debt = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
    'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
cust_credit = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])[
    'AMT_CREDIT_SUM'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})
cust_profile = cust_debt.merge(cust_credit, on=['SK_ID_CURR'], how='left')
cust_profile['DEBT_CREDIT_RATIO'] = cust_profile['TOTAL_CUSTOMER_DEBT'] / cust_profile['TOTAL_CUSTOMER_CREDIT']

del cust_debt
del cust_credit
del cust_profile['TOTAL_CUSTOMER_DEBT'], cust_profile['TOTAL_CUSTOMER_CREDIT']

assert len(list(cust_profile.columns)) == 2

X = X.merge(right=cust_profile, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=cust_profile, how='left', on='SK_ID_CURR')

print('cust_debt_to_credit', X.shape)
del cust_profile

assert X.shape[0] == 307511

#
# cust_overdue_debt
#

bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)
bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

cust_debt = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
    'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
cust_overdue = bureau[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])[
    'AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str,
                                                         columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
cust_profile = cust_debt.merge(cust_overdue, on=['SK_ID_CURR'], how='left')
cust_profile['OVERDUE_DEBT_RATIO'] = cust_profile['TOTAL_CUSTOMER_OVERDUE'] / cust_profile['TOTAL_CUSTOMER_DEBT']

del cust_debt
del cust_overdue
del cust_profile['TOTAL_CUSTOMER_OVERDUE'], cust_profile['TOTAL_CUSTOMER_DEBT']

assert len(list(cust_profile.columns)) == 2

X = X.merge(right=cust_profile, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=cust_profile, how='left', on='SK_ID_CURR')

print('cust_overdue_debt', X.shape)
del cust_profile

assert X.shape[0] == 307511

#
# avg_prolong
#

bureau['CNT_CREDIT_PROLONG'] = bureau['CNT_CREDIT_PROLONG'].fillna(0)
avg_prolong = bureau[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by=['SK_ID_CURR'])[
    'CNT_CREDIT_PROLONG'].mean().reset_index().rename(index=str,
                                                      columns={'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})

assert len(list(avg_prolong.columns)) == 2

X = X.merge(right=avg_prolong, how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_prolong, how='left', on='SK_ID_CURR')

print('avg_prolong', X.shape)
del avg_prolong

assert X.shape[0] == 307511

#
# avg_buro
#

buro_grouped_size = bureau_bal.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
buro_grouped_max = bureau_bal.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
buro_grouped_min = bureau_bal.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()

buro_counts = bureau_bal.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize=False)
buro_counts_unstacked = buro_counts.unstack('STATUS')
buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C',
                                 'STATUS_X', ]
buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max

bureau = bureau.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
del buro_grouped_size
del buro_grouped_max
del buro_grouped_min
del buro_counts
del buro_counts_unstacked
gc.collect()

buro_cat_features = [bcol for bcol in bureau.columns if bureau[bcol].dtype == 'object']
bureau = pd.get_dummies(bureau, columns=buro_cat_features)

avg_buro = bureau.groupby('SK_ID_CURR').mean()
avg_buro.columns = ['SK_ID_BUREAU'] + ['MEAN_OF_{}'.format(c) for c in avg_buro.columns[1:]]
avg_buro['BUREAU_COUNT'] = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']

del avg_buro['SK_ID_BUREAU']

X = X.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

print('avg_buro', X.shape)
del avg_buro

# - POS_CASH_balance.csv
#     - Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
#     - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

#
# max_pos_cash
# avg_pos_cash
# count_pos_cash
#

pos_cash_cat_features = [col for col in pos_cash.columns if pos_cash[col].dtype == 'object']

max_pos_cash = pos_cash[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'SK_DPD', 'SK_DPD_DEF']].groupby(
    'SK_ID_CURR').max()
max_pos_cash.columns = ['MAX_OF_{}'.format(c) for c in max_pos_cash.columns]

avg_pos_cash = pos_cash[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'SK_DPD', 'SK_DPD_DEF']].groupby(
    'SK_ID_CURR').mean()
avg_pos_cash.columns = ['MEAN_OF_{}'.format(c) for c in avg_pos_cash.columns]

count_pos_cash = pos_cash[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
count_pos_cash.columns = ['COUNT_OF_{}'.format(c) for c in count_pos_cash.columns]

X = X.merge(right=max_pos_cash.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=max_pos_cash.reset_index(), how='left', on='SK_ID_CURR')
print('max_pos_cash', X.shape)

X = X.merge(right=avg_pos_cash.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_pos_cash.reset_index(), how='left', on='SK_ID_CURR')
print('avg_pos_cash', X.shape)

X = X.merge(right=count_pos_cash.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=count_pos_cash.reset_index(), how='left', on='SK_ID_CURR')
print('count_pos_cash', X.shape)

del max_pos_cash
del avg_pos_cash
del count_pos_cash
gc.collect()

#
# pos_cash
#

le = LabelEncoder()
pos_cash['NAME_CONTRACT_STATUS'] = le.fit_transform(pos_cash['NAME_CONTRACT_STATUS'].astype(str))

nunique_status = pos_cash[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = pos_cash[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

pos_cash['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
pos_cash['MAX_NUNIQUE_STATUS'] = nunique_status2['NAME_CONTRACT_STATUS']

pos_cash.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

mean_pos_cash = pos_cash.groupby('SK_ID_CURR').mean()
mean_pos_cash.columns = ['MEAN_OF_{}'.format(c) for c in mean_pos_cash.columns]
mean_pos_cash = mean_pos_cash.reset_index()

X = X.merge(mean_pos_cash, how='left', on='SK_ID_CURR')
X_test = X_test.merge(mean_pos_cash, how='left', on='SK_ID_CURR')
print('pos_cash', X.shape)

del mean_pos_cash
gc.collect()

#
# avg_credit_card_bal
#

# - credit_card_balance.csv
#     - Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
#     - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

credit_card_cat_features = [col for col in credit_card_bal.columns if credit_card_bal[col].dtype == 'object']
avg_credit_card_bal = credit_card_bal.copy().drop(credit_card_cat_features, axis=1).groupby('SK_ID_CURR').mean()
del avg_credit_card_bal['SK_ID_PREV']
avg_credit_card_bal.columns = ['MEAN_OF_{}'.format(c) for c in avg_credit_card_bal.columns]

X = X.merge(right=avg_credit_card_bal.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_credit_card_bal.reset_index(), how='left', on='SK_ID_CURR')
print('avg_credit_card_bal', X.shape)

del avg_credit_card_bal
gc.collect()

#
# credit_card_bal
#

credit_card_bal['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card_bal['NAME_CONTRACT_STATUS'].astype(str))
nunique_status = credit_card_bal[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
nunique_status2 = credit_card_bal[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

credit_card_bal['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
credit_card_bal['MAX_NUNIQUE_STATUS'] = nunique_status2['NAME_CONTRACT_STATUS']
credit_card_bal.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

avg_credit_card_bal = credit_card_bal.groupby('SK_ID_CURR').mean()
avg_credit_card_bal.columns = ['MEAN_OF_{}'.format(c) for c in avg_credit_card_bal.columns]
avg_credit_card_bal = avg_credit_card_bal.reset_index()

X = X.merge(avg_credit_card_bal, how='left', on='SK_ID_CURR')
X_test = X_test.merge(avg_credit_card_bal, how='left', on='SK_ID_CURR')
print('credit_card_bal', X.shape)

# - previous_application.csv
#     - All previous applications for Home Credit loans of clients who have loans in our sample.
#     - There is one row for each previous application related to loans in our data sample.

#
# avg_prev
#

prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
prev = pd.get_dummies(prev, columns=prev_cat_features)

avg_prev = prev.groupby('SK_ID_CURR').mean()
avg_prev.columns = ['SK_ID_PREV'] + ['MEAN_OF_{}'.format(c) for c in avg_prev.columns[1:]]
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['COUNT_OF_SK_ID_PREV'] = cnt_prev['SK_ID_PREV']

del avg_prev['SK_ID_PREV']

X = X.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
print('avg_prev', X.shape)

# - installments_payments.csv
#     - Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
#     - There is a) one row for every payment that was made plus b) one row each for missed payment.
#     - One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

#
# avg_payments
#

avg_payments = installment_payment.groupby('SK_ID_CURR').mean()
del avg_payments['SK_ID_PREV']
avg_payments.columns = ['MEAN_OF_{}'.format(c) for c in avg_payments.columns]

avg_payments2 = installment_payment.groupby('SK_ID_CURR').max()
del avg_payments2['SK_ID_PREV']
avg_payments2.columns = ['MAX_OF_{}'.format(c) for c in avg_payments2.columns]

avg_payments3 = installment_payment.groupby('SK_ID_CURR').min()
del avg_payments3['SK_ID_PREV']
avg_payments3.columns = ['MIN_OF_{}'.format(c) for c in avg_payments3.columns]

X = X.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
print('avg_payments', X.shape)

X = X.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
print('avg_payments2', X.shape)

X = X.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
X_test = X_test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
print('avg_payments3', X.shape)

del avg_payments
del avg_payments2
del avg_payments3
gc.collect()

#
# Remove features with many missing values
#

print('Removing features with more than 80% missing...')
X_test = X_test[X_test.columns[X.isnull().mean() < 0.85]]
X = X[X.columns[X.isnull().mean() < 0.85]]

#
# Delete customer Id
#

del X['SK_ID_CURR']
del X_test['SK_ID_CURR']

#
# Prepare data
#

data = X.copy().reset_index()
data.columns = ['index'] + ['{}_{}'.format(i, c) for i, c in enumerate(data.columns[1:])]
data['label'] = y
data.head()

col_type = dict()
col_type['ID'] = 'index'
col_type['target'] = 'label'
col_type['features'] = [x for x in data.columns
                        if x not in [col_type['target'], col_type['ID']]]

train, test = train_test_split(
    data,
    test_size=0.33,
    random_state=1,
    stratify=data[col_type['target']])

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

#
# Train with XGBoost
#

param_grid = {
    # Boosting parameters
    'learning_rate': [0.1],
    'n_estimators': [1000],  # this specify the upper bound, we use early stop to find the optimal value

    # Tree-based parameters
    'max_depth': [5],
    'min_child_weight': [1],
    'gamma': [0],
    'subsample': [0.8],
    'colsample_bytree': [0.8],

    # Regulations parameters
    'reg_lambda': [1],
    'reg_alpha': [1],

    # Other parameters
    'scale_pos_weight': [1]
}

param_table = expand_grid.expand_grid(param_grid)

# Find the optimal number of trees for this learning rate
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=True,
    cv_iterations=1,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=1
)

#
# Tune max_depth and min_child_weight
#

param_grid['max_depth'] = range(3, 10, 2)
param_grid['min_child_weight'] = range(1, 6, 2)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

current_max_depth = param_grid['max_depth'][0]
current_min_child_weight = param_grid['min_child_weight'][0]
param_grid['max_depth'] = np.unique([
    np.max([current_max_depth - 1, 1]),
    current_max_depth,
    current_max_depth + 1
])
param_grid['min_child_weight'] = np.unique([
    np.max([current_min_child_weight - 1, 1]),
    current_min_child_weight,
    current_min_child_weight + 1
])

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune gamma
#

param_grid['gamma'] = [x / 10.0 for x in range(0, 15)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune subsample and colsample_bytree
#

# Coarse search
param_grid['subsample'] = [x / 10.0 for x in range(5, 11)]
param_grid['colsample_bytree'] = [x / 10.0 for x in range(5, 11)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

current_subsample = param_grid['subsample'][0]
current_colsample_bytree = param_grid['colsample_bytree'][0]

param_grid['subsample'] = [x / 100.0 for x in range(
    int(current_subsample * 100) - 15,
    np.min([int(current_subsample * 100) + 15, 105]),
    5
)]

param_grid['colsample_bytree'] = [x / 100.0 for x in range(
    int(current_colsample_bytree * 100) - 15,
    np.min([int(current_colsample_bytree * 100) + 15, 105]),
    5
)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune reg_lambda
#
param_grid['reg_lambda'] = [0, 1e-5, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune reg_alpha
#
param_grid['reg_alpha'] = [0, 1e-5, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Final randomized search
#
param_grid['max_depth'] = [8, 9, 10]
param_grid['min_child_weight'] = [1, 2]
param_grid['gamma'] = [1.1, 1.2, 1.3]
param_grid['subsample'] = norm(loc=param_grid['subsample'][0], scale=0.02)
param_grid['colsample_bytree'] = norm(loc=param_grid['colsample_bytree'][0], scale=0.02)
param_grid['reg_lambda'] = norm(loc=param_grid['reg_lambda'][0], scale=0.02)
param_grid['reg_alpha'] = norm(loc=param_grid['reg_alpha'][0], scale=0.0001)

param_table = random_grid.random_grid(
    param_grid,
    random_iter=100,
    random_state=1
)

param_table['reg_lambda'][param_table['reg_lambda'] < 0] = 0
param_table['reg_alpha'][param_table['reg_alpha'] < 0] = 0

param_grid, _ = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune Learning Rate
#
param_grid['learning_rate'] = [0.001, 0.01, 0.05, 0.1, 0.5]
param_grid['n_estimators'] = [1000]

param_table = expand_grid.expand_grid(param_grid)

param_grid, pred = fit_models.fit_xgboost(
    param_grid,
    param_table,
    train,
    col_type,
    find_n_estimator=True,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Final xgboost params
#
best_param_index = param_table["Score_Weighted"].idxmax()

param_grid_xgboost = param_grid
pred_xgboost = pred["Pred_" + str(best_param_index)] \
    .rename('pred_xgboost') \
    .reset_index()

print(param_grid_xgboost)

#
# Train with LightGBM
#
param_grid = {
    # Boosting parameters
    'learning_rate': [0.1],
    'num_boost_round': [2500],  # this specify the upper bound, we use early_stopping_round to find the optimal value
    'boosting': ['gbdt'],

    # Tree-based parameters
    'num_leaves': [80],
    'min_data_in_leaf': [20],
    'max_depth': [-1],
    'max_bin': [255],  # max number of bins that feature values will be bucketed in

    'bagging_fraction': [0.8],
    'feature_fraction': [0.8],

    # Regulations parameters
    'lambda_l1': [1],
    'lambda_l2': [1],

    # Other parameters
    'is_unbalance': [True],
    'scale_pos_weight': [1.0],
    'device': ['cpu']
}

param_table = expand_grid.expand_grid(param_grid)

# Find the optimal number of trees for this learning rate
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=True,
    cv_iterations=1,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune num_leaves and min_data_in_leaf
#

param_grid['num_leaves'] = [2 ** x for x in range(3, 10, 2)]
param_grid['min_child_weight'] = range(1, 22, 3)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

current_num_leaves = param_grid['num_leaves'][0]
current_min_child_weight = param_grid['min_child_weight'][0]
param_grid['num_leaves'] = range(
    int(0.8 * current_num_leaves),
    int(1.2 * current_num_leaves),
    int(np.ceil(0.02 * current_num_leaves))
)

param_grid['min_child_weight'] = np.unique(
    range(
        np.max([current_min_child_weight - 1, 1]),
        current_min_child_weight + 4
    )
)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune max_depth
#
current_num_leaves = param_grid['num_leaves'][0]
current_avg_max_depth = int(np.ceil(np.log(current_num_leaves) / np.log(2)))

param_grid['max_depth'] = range(
    np.maximum(2, current_avg_max_depth - 2),
    current_avg_max_depth + 2
)

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune max_bin
#
param_grid['max_bin'] = [2 ** x for x in range(3, 11)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune bagging_fraction and feature_fraction
#
param_grid['bagging_fraction'] = [x / 10.0 for x in range(5, 11)]
param_grid['feature_fraction'] = [x / 10.0 for x in range(5, 11)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

current_bagging_fraction = param_grid['bagging_fraction'][0]
current_feature_fraction = param_grid['feature_fraction'][0]

param_grid['bagging_fraction'] = [x / 100.0 for x in range(
    int(current_bagging_fraction * 100) - 15,
    np.min([int(current_bagging_fraction * 100) + 15, 105]),
    5
)]

param_grid['feature_fraction'] = [x / 100.0 for x in range(
    int(current_feature_fraction * 100) - 15,
    np.min([int(current_feature_fraction * 100) + 15, 105]),
    5
)]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune lambda_l2
#
param_grid['lambda_l2'] = [0, 1e-5, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune lambda_l1
#
param_grid['lambda_l1'] = [0, 1e-5, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 3, 5, 10, 100]

param_table = expand_grid.expand_grid(param_grid)
param_grid, _ = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=False,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Tune Learning Rate
#
param_grid['learning_rate'] = [0.001, 0.01, 0.05, 0.1, 0.5]
param_grid['num_boost_round'] = [1000]

param_table = expand_grid.expand_grid(param_grid)

param_grid, pred = fit_models.fit_lightgbm(
    param_grid,
    param_table,
    train,
    col_type,
    find_num_boost_round=True,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Final lightGBM parameter and predictions
#
best_param_index = param_table["Score_Weighted"].idxmax()

param_grid_lightgbm = param_grid
pred_lightgbm = pred["Pred_" + str(best_param_index)].rename('pred_lightgbm').reset_index()

print(param_grid_lightgbm)

#
# Stacking - by logistic regression
#
train_stack = pd.merge(pred_xgboost, pred_lightgbm, on=col_type['ID'])
train_stack = pd.merge(train_stack, train.loc[:, [col_type['ID'], col_type['target']]], on=col_type['ID'])

col_type_stack = col_type.copy()
col_type_stack['features'] = [x for x in train_stack.columns if
                              x not in [col_type_stack['target'], col_type_stack['ID']]]

#
# Tune
#
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'tol': [0.0001]
}

param_table = expand_grid.expand_grid(param_grid)

param_grid, pred = fit_models.fit_logistic_regression(
    param_grid,
    param_table,
    train_stack,
    col_type_stack,
    cv_iterations=5,
    cv_folds=5,
    nthread=N_THREADS,
    verbose=0
)

#
# Best params
#
best_param_index = param_table["Score_Weighted"].idxmax()

param_grid_stack_logistic_regression = param_grid
pred_stack_logistic_regression = pred["Pred_" + str(best_param_index)].rename(
    'pred_stack_logistic_regression').reset_index()

auc_xgboost = metrics.roc_auc_score(train_stack[col_type['target']], train_stack.pred_xgboost)
auc_lgbm = metrics.roc_auc_score(train_stack[col_type['target']], train_stack.pred_lightgbm)
auc_stacked = metrics.roc_auc_score(train_stack[col_type['target']],
                                    pred_stack_logistic_regression.pred_stack_logistic_regression)
print("AUC of xgboost: {}".format(auc_xgboost))
print("AUC of LightGBM: {}".format(auc_lgbm))
print("AUC of stacked Logistic Regression: {}".format(auc_stacked))

#
# Full data
#
train = X.copy().reset_index()
train.columns = ['index'] + ['{}_{}'.format(i, c) for i, c in enumerate(train.columns[1:])]
train['label'] = y
train.head()

test = X_test.copy().reset_index()
test.columns = ['index'] + ['{}_{}'.format(i, c) for i, c in enumerate(test.columns[1:])]
test.head()

col_type = dict()
col_type['ID'] = 'index'
col_type['target'] = 'label'
col_type['features'] = [x for x in data.columns
                        if x not in [col_type['target'], col_type['ID']]]

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

#
# Full training
#
test_pred_xgboost = predict_models.predict_xgboost(
    param_grid_xgboost,
    train,
    test,
    col_type,
    nthread=N_THREADS,
)

test_pred_lightgbm = predict_models.predict_lightgbm(
    param_grid_lightgbm,
    train,
    test,
    col_type,
    nthread=N_THREADS,
)

test_stack = pd.merge(test_pred_xgboost, test_pred_lightgbm, on=col_type['ID'])
test_stack = pd.merge(test_stack, test.loc[:, [col_type['ID']]], on=col_type['ID'])

test_pred_stack_logistic_regression = predict_models.predict_logistic_regression(
    param_grid_stack_logistic_regression,
    train_stack,
    test_stack,
    col_type_stack,
    nthread=N_THREADS,
)

auc_xgboost = metrics.roc_auc_score(test[col_type['target']], test_pred_xgboost.pred_xgboost)
auc_lgbm = metrics.roc_auc_score(test[col_type['target']], test_pred_lightgbm.pred_lightgbm)
auc_stacked = metrics.roc_auc_score(test[col_type['target']],
                                    test_pred_stack_logistic_regression.pred_logistic_regression)
print("AUC of xgboost: {}".format(auc_xgboost))
print("AUC of LightGBM: {}".format(auc_lgbm))
print("AUC of stacked Logistic Regression: {}".format(auc_stacked))

#
# Prepare submit
#
submission_file = pd.read_csv(SUBMISSION_FILE)
submission_file.TARGET = test_pred_stack_logistic_regression['pred_logistic_regression']

print(submission_file.head())

output_file_name = os.path.join(SUBMISSION_DIR, 'stacked_{0:%Y-%m-%d_%H:%M:%S}.csv'.format(datetime.now()))
print(output_file_name)
submission_file.to_csv(output_file_name, index=False)
