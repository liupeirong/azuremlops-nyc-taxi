import numpy as np
import joblib
from azureml.core.model import Model
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type \
    import NumpyParameterType
from azureml.monitoring import ModelDataCollector
import consts


def init():
    global model
    global inputs_dc, prediction_dc
    model_path = Model.get_model_path(consts.model_name)
    model = joblib.load(model_path)
    inputs_dc = ModelDataCollector(
            consts.model_name,
            designation="inputs",
            feature_names=[
                'vendorID',
                'passengerCount',
                'tripDistance',
                'pickupLongitude',
                'pickupLatitude',
                'dropoffLongitude',
                'dropoffLatitude',
                'totalAmount',
                'month_num',
                'day_of_month',
                'day_of_week',
                'hour_of_day'])
    prediction_dc = ModelDataCollector(
            consts.model_name,
            designation="predictions",
            feature_names=["duration"])


# input is an array of datapoints, each has an array of features
input_sample = np.array([
    [1, 1, 1.00, -73.957909, 40.670761,
     -73.952194, 40.662312, 8.15, 1, 17, 5, 1]])
# output is an array of predictions
output_sample = np.array([7.81327569])


@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        inputs_dc.collect(data)
        prediction_dc.collect(result)
        return result.tolist()
    except Exception as e:
        error = str(e)
        inputs_dc.collect(data)
        prediction_dc.collect(error)
        return error
