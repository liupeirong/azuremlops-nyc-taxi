# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from azureml.core.model import Model
import consts

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type \
    import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type \
    import PandasParameterType


input_sample = pd.DataFrame(data=[
    {'vendorID': 2.0, 'passengerCount': 1.0, 'tripDistance': 0.43,
     'pickupLongitude': -73.9254608154, 'pickupLatitude': 40.8323936462,
     'dropoffLongitude': -73.9238586426, 'dropoffLatitude': 40.8364105225,
     'totalAmount': 4.8, 'month_num': 1.0, 'day_of_month': 30.0,
     'day_of_week': 4.0, 'hour_of_day': 21.0}])
output_sample = np.array([0])


def init():
    global model
    global scoring_model

    model_path = Model.get_model_path(model_name=consts.model_name_automl)
    scoring_model_path = Model.get_model_path(model_name='scoring_explainer')
    model = joblib.load(model_path)
    scoring_model = joblib.load(scoring_model_path)


@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        local_importance_values = scoring_model.explain(data)
        return {'result': result.tolist(),
                'local_importance_values': local_importance_values}
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
