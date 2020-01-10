import os
import argparse
import joblib
import lightgbm as lgb
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from azureml.core import Run
import utils
import consts


def split_data(df):
    y_df = df.duration
    x_df = df.drop(columns='duration')
    x_train, x_test, y_train, y_test = \
        train_test_split(x_df, y_df, test_size=0.2, random_state=223)
    return x_train, x_test, y_train, y_test


def train_model(x_train, x_test, y_train, y_test):
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        # metric 12: mean_squared_error, 11: mean_absolute_error
        'metric': {'l2', 'l1'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20,
                    valid_sets=lgb_eval,
                    early_stopping_rounds=5,
                    categorical_feature=[
                        'vendorID',
                        'month_num',
                        'day_of_month',
                        'day_of_week',
                        'hour_of_day'])

    # evaluate the model
    y_predict = gbm.predict(x_test)
    rmse = math.sqrt(mean_squared_error(y_true=y_test, y_pred=y_predict))
    mape = utils.MAPE(y_test, y_predict)

    return gbm, rmse, mape


def main():
    # find where data is
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_folder',
        type=str,
        default='',
        dest='data_folder',
        help='folder in datastore where training data is located')
    args = parser.parse_args()

    run = Run.get_context()
    run.tag('data_folder',
            utils.last_two_folders_if_exists(args.data_folder))

    # read and process data
    # df = utils.read_raw_data(data_folder)
    # df = utils.process_raw_data(df)
    df = utils.read_train_data(args.data_folder)
    x_train, x_test, y_train, y_test = split_data(df)

    model, rmse, mape = train_model(x_train, x_test, y_train, y_test)
    run.log('rmse', rmse)
    run.log('mape', mape)

    # save the model
    os.makedirs('outputs', exist_ok=True)
    model_file = os.path.join('outputs', consts.model_name)
    run.tag('model_file', model_file)
    joblib.dump(value=model, filename=model_file)


if __name__ == '__main__':
    main()
