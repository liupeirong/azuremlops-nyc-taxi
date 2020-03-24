import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path

min_latitude = 40.53
max_latitude = 40.88
min_longitude = -74.09
max_longitude = -73.72
min_distance = 0.25
max_distance = 31
max_duration = 180  # minutes


def build_features(s):
    """
    Build additional features for the input row of a dataframe.
    Returns the same row with additional feature columns.
    """
    s['month_num'] = s['lpepPickupDatetime'].month
    s['day_of_month'] = s['lpepPickupDatetime'].day
    s['day_of_week'] = s['lpepPickupDatetime'].weekday()
    s['hour_of_day'] = s['lpepPickupDatetime'].hour
    s['duration'] = (
        (s['lpepDropoffDatetime'] - s['lpepPickupDatetime']).seconds // 60)
    return s


def read_raw_data(data_folder_or_file):
    """
    Read all csv files from data/input folder and concat to
    return a single dataframe
    """
    if os.path.isfile(data_folder_or_file):
        print("Reading raw data file {}".format(data_folder_or_file))
        df = pd.read_csv(
            data_folder_or_file,
            index_col=0,
            parse_dates=['lpepPickupDatetime', 'lpepDropoffDatetime'],
            infer_datetime_format=True)
    else:
        print("Reading files in raw data folder {}"
              .format(data_folder_or_file))
        all_files = glob.glob(os.path.join(data_folder_or_file, '**/*.csv'),
                              recursive=True)
        df_list = []
        for f in all_files:
            df_from_each_file = pd.read_csv(
                f,
                index_col=0,
                parse_dates=['lpepPickupDatetime', 'lpepDropoffDatetime'],
                infer_datetime_format=True)
            df_list.append(df_from_each_file)

        df = pd.concat(df_list)
    return df


def process_raw_data(df):
    """
    Fileter raw data to valid range,
    build additional features,
    filter again additional features to valid range,
    remove unused columns.
    """
    df = df.loc[
            (df.pickupLatitude >= min_latitude) &
            (df.pickupLatitude <= max_latitude) &
            (df.pickupLongitude >= min_longitude) &
            (df.pickupLongitude <= max_longitude) &
            (df.tripDistance >= min_distance) &
            (df.tripDistance < max_distance) &
            (df.passengerCount > 0) &
            (df.totalAmount > 0)]
    pd.options.mode.chained_assignment = None
    df['month_num'] = df.lpepPickupDatetime.dt.month
    df['day_of_month'] = df.lpepPickupDatetime.dt.day
    df['day_of_week'] = df.lpepPickupDatetime.dt.weekday
    df['hour_of_day'] = df.lpepPickupDatetime.dt.hour
    df['duration'] = (
        (df.lpepDropoffDatetime - df.lpepPickupDatetime).dt.seconds // 60)
    df = df.loc[(df.duration < max_duration)]

    columns_to_remove = [
        "lpepPickupDatetime", "lpepDropoffDatetime", "puLocationId",
        "doLocationId", "extra", "mtaTax", "improvementSurcharge",
        "tollsAmount", "ehailFee", "tripType", "rateCodeID",
        "storeAndFwdFlag", "paymentType", "fareAmount", "tipAmount"]
    for col in columns_to_remove:
        df.pop(col)

    return df


def write_train_data(df, file_path, file_name):
    """
    Read all csv files from input folder and concat to return a single
    dataframe. To detect data drift, the input data to web service doesn't
    have an index column, if the training data has index column, it will
    cause mismatch error. So avoid writing index column.
    """
    Path(file_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(path_or_buf=os.path.join(file_path, file_name), index=False)
    return


def read_train_data(data_folder):
    """
    Read all csv files from input folder and concat to return a single
    dataframe. Training data doesn't include index column.
    """
    all_files = glob.glob(os.path.join(data_folder, '**/*.csv'),
                          recursive=True)
    df_from_each_file = (pd.read_csv(f, index_col=None) for f in all_files)
    df = pd.concat(df_from_each_file)
    return df


def MAPE(actual, pred):
    """
    Calculate mean absolute percentage error.
    Remove NA and values where actual is close to zero.
    """
    not_na = ~(np.isnan(actual) | np.isnan(pred))
    not_zero = ~np.isclose(actual, 0.0)
    actual_safe = actual[not_na & not_zero]
    pred_safe = pred[not_na & not_zero]
    APE = 100*np.abs((actual_safe - pred_safe)/actual_safe)
    return np.mean(APE)


def last_two_folders_if_exists(path):
    last_slash = path.rfind('/')
    if last_slash != -1:
        second_last_slash = path.rfind('/', 0, last_slash)
        if second_last_slash != -1:
            return path[second_last_slash:]
        else:
            return path[last_slash:]
    else:
        return path
