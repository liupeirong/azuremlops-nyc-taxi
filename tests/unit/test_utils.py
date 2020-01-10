import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), 'code'))
import utils  # noqa: E402


def test_read_raw_data():
    raw_data_dir = 'tests/unit/test_data/raw'
    df = utils.read_raw_data(raw_data_dir)
    assert len(df) == 2, "test data should have 2 records"
    assert df.dtypes['lpepPickupDatetime'].type == np.datetime64, \
        "pickupDatetime should read in as datetime type"
    assert df.dtypes['lpepDropoffDatetime'].type == np.datetime64, \
        "dropoffDatetime should read in as datetime type"
    assert len(df.columns) == 23, "raw data should have 23 columns"


def test_process_raw_data():
    raw_data_dir = 'tests/unit/test_data/raw'
    train_data_dir = 'tests/unit/test_data/processed'
    dfraw = utils.read_raw_data(raw_data_dir)
    dfuut = utils.process_raw_data(dfraw)
    dfexpected = utils.read_train_data(train_data_dir)
    np.array_equal(dfuut.values, dfexpected.values)
