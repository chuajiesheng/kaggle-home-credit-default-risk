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
N_THREADS = round(cpu_count() * 2 * 0.9)

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
print('finished reading file')

#
# Feature Generation
#

categorical_features = [col for col in X.columns if X[col].dtype == 'object']

train_test = pd.concat([X, X_test])
train_test = pd.get_dummies(train_test, columns=categorical_features)

X = train_test.iloc[:X.shape[0], :]
X_test = train_test.iloc[X.shape[0]:, ]


def gen_relative_calculation(train_test_df):
    source_features = ['DAYS_EMPLOYED', 'DAYS_BIRTH', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'CNT_FAM_MEMBERS', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL']
    application_df = train_test_df[['SK_ID_CURR'] + source_features].copy()

    application_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    application_df['DAYS_EMPLOYED_PERC'] = application_df['DAYS_EMPLOYED'] / application_df['DAYS_BIRTH']
    application_df['INCOME_CREDIT_PERC'] = application_df['AMT_INCOME_TOTAL'] / application_df['AMT_CREDIT']
    application_df['INCOME_PER_PERSON'] = application_df['AMT_INCOME_TOTAL'] / application_df['CNT_FAM_MEMBERS']
    application_df['ANNUITY_INCOME_PERC'] = application_df['AMT_ANNUITY'] / application_df['AMT_INCOME_TOTAL']

    for f in source_features:
        del application_df[f]

    return application_df


with gen_relative_calculation(train_test) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('gen_relative_calculation', X.shape)
    assert X.shape[0] == 307511
    gc.collect()
#
# Generated features from previous application
#


def gen_prev_installment_feature(installment_payment_df):
    sorted_prev = installment_payment_df[['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']].sort_values(
        ['SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
    compare_to_last = sorted_prev.groupby(by=['SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].diff().fillna(1)
    sorted_prev['COMPARED_TO_LAST'] = compare_to_last
    std_of_installment_seq = sorted_prev[['SK_ID_PREV', 'COMPARED_TO_LAST']].groupby(by=['SK_ID_PREV']).std().reset_index()
    std_of_installment_seq = std_of_installment_seq.rename(index=str,
                                                           columns={'COMPARED_TO_LAST': 'STD_OF_INSTALLMENT_SEQ'})
    prev_installment_feature = installment_payment_df[['SK_ID_CURR', 'SK_ID_PREV']].copy()
    prev_installment_feature = prev_installment_feature.merge(right=std_of_installment_seq, how='left', on='SK_ID_PREV')
    late_installment = installment_payment_df[
        ['SK_ID_CURR', 'SK_ID_PREV', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT']].sort_values(['SK_ID_CURR', 'SK_ID_PREV'])
    late_installment['LATE'] = late_installment['DAYS_INSTALMENT'] - late_installment['DAYS_ENTRY_PAYMENT']
    late_mean = late_installment[['SK_ID_PREV', 'LATE']].groupby(by=['SK_ID_PREV']).mean().fillna(0).reset_index()
    late_mean = late_mean.rename(index=str, columns={'LATE': 'MEAN_OF_LATE_INSTALLMENT'})
    prev_installment_feature = prev_installment_feature.merge(right=late_mean, how='left', on='SK_ID_PREV')
    pay_less = installment_payment_df[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_INSTALMENT', 'AMT_PAYMENT']].sort_values(
        ['SK_ID_CURR', 'SK_ID_PREV'])
    pay_less['INSUFFICIENT_PAYMENT'] = pay_less['AMT_INSTALMENT'] - pay_less['AMT_PAYMENT']
    pay_less = pay_less[['SK_ID_PREV', 'INSUFFICIENT_PAYMENT']].groupby(by=['SK_ID_PREV']).mean().fillna(0).reset_index()
    pay_less = pay_less.rename(index=str, columns={'INSUFFICIENT_PAYMENT': 'MEAN_OF_INSUFFICIENT_PAYMENT'})
    prev_installment_feature = prev_installment_feature.merge(right=pay_less, how='left', on='SK_ID_PREV')
    prev_installment_feature_by_curr = prev_installment_feature.groupby(by=['SK_ID_CURR']).mean().fillna(0).reset_index()
    del prev_installment_feature_by_curr['SK_ID_PREV']

    return prev_installment_feature_by_curr


with gen_prev_installment_feature(installment_payment) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('prev_installment_feature', X.shape)
    assert X.shape[0] == 307511
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

def gen_bur_month_balance(bureau_df, bureau_bal_df):
    agg_by = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    bureau_bal_agg = bureau_bal_df.copy().groupby('SK_ID_BUREAU').agg(agg_by)
    bureau_bal_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_bal_agg.columns.tolist()])

    bureau_agg = bureau_df.copy().join(bureau_bal_agg, how='left', on='SK_ID_BUREAU')
    bureau_agg.drop(columns='SK_ID_BUREAU', inplace=True)

    agg_by = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'AMT_ANNUITY': ['max', 'mean'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    del bureau_agg['CREDIT_ACTIVE'], bureau_agg['CREDIT_CURRENCY'], bureau_agg['CREDIT_TYPE']
    bureau_agg = bureau_agg.groupby('SK_ID_CURR').agg(agg_by)
    bureau_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    return bureau_agg.reset_index()


with gen_bur_month_balance(bureau, bureau_bal) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('bur_month_balance', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


#
# credit_variety
#


def gen_credit_variety(bureau_df):
    count_day_credit = bureau_df[['SK_ID_CURR', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])['DAYS_CREDIT'].count()
    count_day_credit = count_day_credit.reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})

    count_credit_type = bureau_df[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by=['SK_ID_CURR'])['CREDIT_TYPE'].nunique()
    count_credit_type = count_credit_type.reset_index().rename(index=str, columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})

    credit_variety = count_day_credit.merge(right=count_credit_type, how='left', on='SK_ID_CURR')
    credit_variety['AVERAGE_LOAN_TYPE'] = credit_variety['BUREAU_LOAN_COUNT'] / credit_variety['BUREAU_LOAN_TYPES']

    return credit_variety


with gen_credit_variety(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('credit_variety', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


#
# bureau_active
#


def gen_bureau_active(bureau_df):
    def count_active(x):
        return 0 if x == 'Closed' else 1

    active = bureau_df[['SK_ID_CURR', 'CREDIT_ACTIVE']].apply(lambda x: count_active(x.CREDIT_ACTIVE), axis=1)
    bureau_active = bureau_df[['SK_ID_CURR']].copy()
    bureau_active['ACTIVE_COUNT'] = active
    bureau_active_sum = bureau_active.groupby(by=['SK_ID_CURR'])['ACTIVE_COUNT'].sum().reset_index()

    bureau_active_mean = bureau_active.groupby(by=['SK_ID_CURR'])['ACTIVE_COUNT'].mean().reset_index()
    bureau_active_mean = bureau_active_mean.rename(index=str, columns={'ACTIVE_COUNT': 'ACTIVE_LOANS_PERCENTAGE'})

    return bureau_active_sum.merge(right=bureau_active_mean, how='left', on='SK_ID_CURR')


with gen_bureau_active(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('bureau_active', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# day_credit_group
#


def gen_day_credit_group(bureau_df):
    day_credit_group = bureau_df[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby(by=['SK_ID_CURR'])
    day_credit_group = day_credit_group.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending=False)).reset_index(
        drop=True)
    day_credit_group['DAYS_CREDIT1'] = day_credit_group['DAYS_CREDIT'] * -1
    day_credit_group['DAYS_DIFF'] = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
    day_credit_group['DAYS_DIFF'] = day_credit_group['DAYS_DIFF'].fillna(0).astype('uint32')
    del day_credit_group['DAYS_CREDIT1'], day_credit_group['DAYS_CREDIT'], day_credit_group['SK_ID_BUREAU']

    day_credit_group_mean = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_DIFF'].mean()
    day_credit_group_mean = day_credit_group_mean.reset_index().rename(index=str, columns={'DAYS_DIFF': 'MEAN_DAYS_DIFF'})
    day_credit_group_max = day_credit_group.groupby(by=['SK_ID_CURR'])['DAYS_DIFF'].max()
    day_credit_group_max = day_credit_group_max.reset_index().rename(index=str, columns={'DAYS_DIFF': 'MAX_DAYS_DIFF'})

    return day_credit_group_mean.merge(right=day_credit_group_max, how='left', on='SK_ID_CURR')


with gen_day_credit_group(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('day_credit_group', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# bureau_credit_time
#


def gen_bureau_credit_time(bureau_df):
    def check_credit_time(x):
        return 0 if x < 0 else 1

    credit_time = bureau_df[['SK_ID_CURR', 'DAYS_CREDIT_ENDDATE']].apply(lambda x: check_credit_time(x.DAYS_CREDIT_ENDDATE),
                                                                         axis=1)
    bureau_credit_time = bureau_df[['SK_ID_CURR']].copy()
    bureau_credit_time['CREDIT_TIME'] = credit_time

    credit_time_mean = bureau_credit_time.groupby(by=['SK_ID_CURR'])['CREDIT_TIME'].mean()
    credit_time_mean = credit_time_mean.reset_index().rename(index=str, columns={'CREDIT_TIME': 'MEAN_CREDIT_TIME'})

    credit_time_max = bureau_credit_time.groupby(by=['SK_ID_CURR'])['CREDIT_TIME'].max()
    credit_time_max = credit_time_max.reset_index().rename(index=str, columns={'CREDIT_TIME': 'MAX_CREDIT_TIME'})

    return credit_time_mean.merge(right=credit_time_max, how='left', on='SK_ID_CURR')


with gen_bureau_credit_time(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('bureau_credit_time', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# loan_count
#


def gen_loan_count(bureau_df):
    positive_credit_end_date = bureau_df[bureau_df['DAYS_CREDIT_ENDDATE'] > 0]
    max_per_loan = positive_credit_end_date.groupby(by=['SK_ID_CURR', 'SK_ID_BUREAU'])[
        'DAYS_CREDIT_ENDDATE'].max().reset_index()
    max_per_loan = max_per_loan.rename(index=str, columns={'DAYS_CREDIT_ENDDATE': 'MAX_DAYS_CREDIT_ENDDATE'})
    max_per_user = max_per_loan.groupby(by=['SK_ID_CURR'])['MAX_DAYS_CREDIT_ENDDATE'].max().reset_index()

    current_loan_count = positive_credit_end_date.groupby(by=['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index()
    current_loan_count = current_loan_count.rename(index=str, columns={'SK_ID_BUREAU': 'COUNT_SK_ID_BUREAU'})

    return max_per_user.merge(right=current_loan_count, how='left', on='SK_ID_CURR')


with gen_loan_count(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('loan_count', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# cust_debt_to_credit
#


def gen_cust_debt_to_credit(bureau_df):
    bureau_df['AMT_CREDIT_SUM_DEBT'] = bureau_df['AMT_CREDIT_SUM_DEBT'].fillna(0)
    bureau_df['AMT_CREDIT_SUM'] = bureau_df['AMT_CREDIT_SUM'].fillna(0)
    cust_debt = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    cust_credit = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM': 'TOTAL_CUSTOMER_CREDIT'})
    cust_profile = cust_debt.merge(cust_credit, on=['SK_ID_CURR'], how='left')
    cust_profile['DEBT_CREDIT_RATIO'] = cust_profile['TOTAL_CUSTOMER_DEBT'] / cust_profile['TOTAL_CUSTOMER_CREDIT']

    del cust_profile['TOTAL_CUSTOMER_DEBT'], cust_profile['TOTAL_CUSTOMER_CREDIT']
    assert len(list(cust_profile.columns)) == 2

    return cust_profile


with gen_cust_debt_to_credit(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('cust_debt_to_credit', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# cust_overdue_debt
#


def gen_cust_overdue_debt(bureau_df):
    bureau_df['AMT_CREDIT_SUM_DEBT'] = bureau_df['AMT_CREDIT_SUM_DEBT'].fillna(0)
    bureau_df['AMT_CREDIT_SUM_OVERDUE'] = bureau_df['AMT_CREDIT_SUM_OVERDUE'].fillna(0)
    cust_debt = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'TOTAL_CUSTOMER_DEBT'})
    cust_overdue = bureau_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])[
        'AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str,
                                                             columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})
    cust_profile = cust_debt.merge(cust_overdue, on=['SK_ID_CURR'], how='left')
    cust_profile['OVERDUE_DEBT_RATIO'] = cust_profile['TOTAL_CUSTOMER_OVERDUE'] / cust_profile['TOTAL_CUSTOMER_DEBT']

    del cust_profile['TOTAL_CUSTOMER_OVERDUE'], cust_profile['TOTAL_CUSTOMER_DEBT']
    assert len(list(cust_profile.columns)) == 2

    return cust_profile


with gen_cust_overdue_debt(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('cust_overdue_debt', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# avg_prolong
#


def gen_avg_prolong(bureau_df):
    global avg_prolong
    bureau_df['CNT_CREDIT_PROLONG'] = bureau_df['CNT_CREDIT_PROLONG'].fillna(0)
    avg_prolong = bureau_df[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by=['SK_ID_CURR'])[
        'CNT_CREDIT_PROLONG'].mean().reset_index().rename(index=str,
                                                          columns={'CNT_CREDIT_PROLONG': 'AVG_CREDITDAYS_PROLONGED'})
    assert len(list(avg_prolong.columns)) == 2


with gen_avg_prolong(bureau) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('avg_prolong', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# avg_buro
#


def gen_avg_buro(bureau_df, bureau_bal_df):
    buro_counts = bureau_bal_df.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize=False)
    buro_counts_unstacked = buro_counts.unstack('STATUS')
    buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1', 'STATUS_2', 'STATUS_3', 'STATUS_4', 'STATUS_5', 'STATUS_C',
                                     'STATUS_X', ]

    bureau_df = bureau_df.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
    buro_cat_features = [bcol for bcol in bureau_df.columns if bureau_df[bcol].dtype == 'object']
    bureau_df = pd.get_dummies(bureau_df, columns=buro_cat_features)

    avg_buro = bureau_df.groupby('SK_ID_CURR').mean()
    avg_buro.columns = ['SK_ID_BUREAU'] + ['MEAN_OF_{}'.format(c) for c in avg_buro.columns[1:]]
    avg_buro['BUREAU_COUNT'] = bureau_df[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']

    del avg_buro['SK_ID_BUREAU']
    return avg_buro


with gen_avg_buro(bureau, bureau_bal) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('avg_buro', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


# - POS_CASH_balance.csv
#     - Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
#     - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

#
# max_pos_cash
# avg_pos_cash
# count_pos_cash
#


def gen_pos_cash_features(pos_cash_df):
    max_pos_cash = pos_cash_df[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'SK_DPD', 'SK_DPD_DEF']].groupby(
        'SK_ID_CURR').max()
    max_pos_cash.columns = ['MAX_OF_{}'.format(c) for c in max_pos_cash.columns]
    avg_pos_cash = pos_cash_df[['SK_ID_CURR', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'SK_DPD', 'SK_DPD_DEF']].groupby(
        'SK_ID_CURR').mean()
    avg_pos_cash.columns = ['MEAN_OF_{}'.format(c) for c in avg_pos_cash.columns]
    count_pos_cash = pos_cash_df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    count_pos_cash.columns = ['COUNT_OF_{}'.format(c) for c in count_pos_cash.columns]

    return max_pos_cash.merge(right=avg_pos_cash, how='left', on='SK_ID_CURR').merge(right=count_pos_cash, how='left', on='SK_ID_CURR')


with gen_pos_cash_features(pos_cash) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('pos_cash_features', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# pos_cash
#


def gen_agg_pos_cash(pos_cash_df):
    pos_cash_df = pos_cash_df.copy()

    agg_by = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }

    pos_cash_agg = pos_cash_df.copy().groupby('SK_ID_CURR').agg(agg_by)
    pos_cash_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_cash_agg.columns.tolist()])
    return pos_cash_agg.reset_index()


with gen_agg_pos_cash(pos_cash) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('agg_pos_cash', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


def gen_mean_pos_cash(pos_cash_df):
    le = LabelEncoder()
    pos_cash_df['NAME_CONTRACT_STATUS'] = le.fit_transform(pos_cash_df['NAME_CONTRACT_STATUS'].astype(str))

    nunique_status = pos_cash_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = pos_cash_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

    pos_cash_df['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    pos_cash_df['MAX_NUNIQUE_STATUS'] = nunique_status2['NAME_CONTRACT_STATUS']
    pos_cash_df.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

    mean_pos_cash = pos_cash_df.groupby('SK_ID_CURR').mean()
    mean_pos_cash.columns = ['MEAN_OF_{}'.format(c) for c in mean_pos_cash.columns]

    return mean_pos_cash.reset_index()


with gen_mean_pos_cash(pos_cash) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('pos_cash', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# avg_credit_card_bal
#

# - credit_card_balance.csv
#     - Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
#     - This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.


def gen_avg_credit_card_bal(credit_card_bal_df):
    credit_card_cat_features = [col for col in credit_card_bal_df.columns if credit_card_bal_df[col].dtype == 'object']
    avg_credit_card_bal = credit_card_bal_df.copy().drop(credit_card_cat_features, axis=1).groupby('SK_ID_CURR').mean()
    del avg_credit_card_bal['SK_ID_PREV']
    avg_credit_card_bal.columns = ['MEAN_OF_{}'.format(c) for c in avg_credit_card_bal.columns]
    return avg_credit_card_bal


with gen_avg_credit_card_bal(credit_card_bal) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('avg_credit_card_bal', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

#
# credit_card_bal
#


def gen_agg_credit_card_bal(credit_card_bal_df):
    credit_card_bal_agg = credit_card_bal_df.groupby('SK_ID_CURR').agg(['min', 'max', 'sum', 'var'])
    credit_card_bal_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in credit_card_bal_agg.columns.tolist()])
    credit_card_bal_agg['CC_COUNT'] = credit_card_bal.groupby('SK_ID_CURR').size()

    return credit_card_bal_agg.reset_index()


with gen_agg_credit_card_bal(credit_card_bal) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('agg_credit_card_bal', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


def gen_credit_card_bal(credit_card_bal_df):
    le = LabelEncoder()
    credit_card_bal_df['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card_bal_df['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = credit_card_bal_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = credit_card_bal_df[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()

    credit_card_bal_df['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    credit_card_bal_df['MAX_NUNIQUE_STATUS'] = nunique_status2['NAME_CONTRACT_STATUS']
    credit_card_bal_df.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)

    avg_credit_card_bal = credit_card_bal_df.groupby('SK_ID_CURR').mean()
    avg_credit_card_bal.columns = ['MEAN_OF_{}'.format(c) for c in avg_credit_card_bal.columns]
    return avg_credit_card_bal.reset_index()


with gen_credit_card_bal(credit_card_bal) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('credit_card_bal', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


# - previous_application.csv
#     - All previous applications for Home Credit loans of clients who have loans in our sample.
#     - There is one row for each previous application related to loans in our data sample.

#
# avg_prev
#

def gen_agg_prev(prev_df):
    prev_df = prev_df.copy()
    prev_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    prev_df['APP_CREDIT_PERC'] = prev_df['AMT_APPLICATION'] / prev_df['AMT_CREDIT']

    agg_by = {
        'AMT_ANNUITY': ['min', 'max'],
        'AMT_APPLICATION': ['min', 'max'],
        'AMT_CREDIT': ['min', 'max'],
        'APP_CREDIT_PERC': ['min', 'max', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max'],
        'AMT_GOODS_PRICE': ['min', 'max'],
        'HOUR_APPR_PROCESS_START': ['min', 'max'],
        'RATE_DOWN_PAYMENT': ['min', 'max'],
        'DAYS_DECISION': ['min', 'max'],
        'CNT_PAYMENT': ['sum'],
    }

    prev_agg = prev_df.copy().groupby('SK_ID_CURR').agg(agg_by)
    prev_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    return prev_agg.reset_index()


with gen_agg_prev(prev) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('agg_prev', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


def gen_avg_prev(prev_df):
    prev_cat_features = [pcol for pcol in prev_df.columns if prev_df[pcol].dtype == 'object']
    prev_df = pd.get_dummies(prev_df, columns=prev_cat_features)

    avg_prev = prev_df.groupby('SK_ID_CURR').mean()
    avg_prev.columns = ['SK_ID_PREV'] + ['MEAN_OF_{}'.format(c) for c in avg_prev.columns[1:]]
    cnt_prev = prev_df[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    avg_prev['COUNT_OF_SK_ID_PREV'] = cnt_prev['SK_ID_PREV']
    del avg_prev['SK_ID_PREV']

    return avg_prev


with gen_avg_prev(prev) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('avg_prev', X.shape)
    assert X.shape[0] == 307511
    gc.collect()

# - installments_payments.csv
#     - Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
#     - There is a) one row for every payment that was made plus b) one row each for missed payment.
#     - One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

#
# avg_payments
#


def gen_agg_installments(installment_payment_df):
    installment_payment_agg = installment_payment_df.copy()

    installment_payment_agg['PAYMENT_PERC'] = installment_payment_agg['AMT_PAYMENT'] / installment_payment_agg['AMT_INSTALMENT']
    installment_payment_agg['PAYMENT_DIFF'] = installment_payment_agg['AMT_INSTALMENT'] - installment_payment_agg['AMT_PAYMENT']
    installment_payment_agg['DPD'] = installment_payment_agg['DAYS_ENTRY_PAYMENT'] - installment_payment_agg['DAYS_INSTALMENT']
    installment_payment_agg['DBD'] = installment_payment_agg['DAYS_INSTALMENT'] - installment_payment_agg['DAYS_ENTRY_PAYMENT']
    installment_payment_agg['DPD'] = installment_payment_agg['DPD'].apply(lambda x: x if x > 0 else 0)
    installment_payment_agg['DBD'] = installment_payment_agg['DBD'].apply(lambda x: x if x > 0 else 0)

    agg_by = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['sum'],
        'DBD': ['sum'],
        'PAYMENT_PERC': ['sum', 'var'],
        'PAYMENT_DIFF': ['sum', 'var'],
        'AMT_INSTALMENT': ['sum'],
        'AMT_PAYMENT': ['sum'],
        'DAYS_ENTRY_PAYMENT': ['sum']
    }

    installment_payment_agg = installment_payment_agg.groupby('SK_ID_CURR').agg(agg_by)
    installment_payment_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in installment_payment_agg.columns.tolist()])
    return installment_payment_agg.reset_index()


with gen_agg_installments(installment_payment) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('agg_installments', X.shape)
    assert X.shape[0] == 307511
    gc.collect()


def gen_avg_payments(installment_payment_df):
    avg_payments = installment_payment_df.groupby('SK_ID_CURR').mean()
    del avg_payments['SK_ID_PREV']
    avg_payments.columns = ['MEAN_OF_{}'.format(c) for c in avg_payments.columns]

    avg_payments2 = installment_payment_df.groupby('SK_ID_CURR').max()
    del avg_payments2['SK_ID_PREV']
    avg_payments2.columns = ['MAX_OF_{}'.format(c) for c in avg_payments2.columns]

    avg_payments3 = installment_payment_df.groupby('SK_ID_CURR').min()
    del avg_payments3['SK_ID_PREV']
    avg_payments3.columns = ['MIN_OF_{}'.format(c) for c in avg_payments3.columns]

    avg_payments = avg_payments.reset_index()
    avg_payments2 = avg_payments2.reset_index()
    avg_payments3 = avg_payments3.reset_index()

    return avg_payments.merge(right=avg_payments2, how='left', on='SK_ID_CURR').merge(right=avg_payments3, how='left', on='SK_ID_CURR')


with gen_avg_payments(installment_payment) as df:
    X = X.merge(right=df, how='left', on='SK_ID_CURR')
    X_test = X_test.merge(right=df, how='left', on='SK_ID_CURR')
    print('avg_payments', X.shape)
    assert X.shape[0] == 307511
    gc.collect()
