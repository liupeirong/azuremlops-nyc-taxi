import os
import argparse
import logging
import aml_utils as amlutils
from azureml.train.automl import AutoMLConfig
from azureml.automl.core.featurization import FeaturizationConfig
from azureml.train.automl.runtime import AutoMLStep
from azureml.core import Dataset
from azureml.pipeline.core import Pipeline, PipelineEndpoint

CODE_PATH = '../code'
TRAIN_STEP_NAME = 'nyc_automl_regression'
PIPELINE_NAME = 'NYCtaxi_pipeline_test'
PIPELINE_ENDPOINT_NAME = PIPELINE_NAME + '_endpoint'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--training_data_file',
        type=str,
        default='201501.csv',
        dest='training_data_file',
        help='path in the datastore where training data is located')
    parser.add_argument(
        '--build_id',
        type=str,
        default='manual',
        dest='build_id',
        help='build id from ci/cd pipelines')

    args = parser.parse_args()
    # there's no good way to pass pipeline parameters to automl config
    # at the moment
    automl_settings = {
        'iterations': 20,
        'iteration_timeout_minutes': 2,
        'experiment_timeout_minutes': 20,
        'whitelist_models': ['LightGBM'],
        'primary_metric': 'normalized_root_mean_squared_error',
        'n_cross_validations': 5,
    }
    ws, ds, compute_target, _ = amlutils.setup_azureml()
    pipeline = create_automl_pipeline(
        ws, ds, compute_target,
        args.training_data_file, automl_settings)
    publish_automl_pipeline(ws, pipeline, args.build_id)


def create_automl_pipeline(ws, ds, compute_target,
                           training_data_file, automl_settings):
    tabular_train_dataset = Dataset.Tabular.from_delimited_files(
        path=[(ds, os.path.join('train', training_data_file))])

    automl_config = build_automl_config(
        False, automl_settings, tabular_train_dataset, compute_target)
    trainWithAutoMLStep = AutoMLStep(
        name=TRAIN_STEP_NAME,
        automl_config=automl_config,
        allow_reuse=False)
    pipeline_steps = [trainWithAutoMLStep]
    pipeline = Pipeline(workspace=ws, steps=pipeline_steps)
    pipeline._set_experiment_name
    pipeline.validate()
    print("Pipeline is built.")
    return pipeline


def publish_automl_pipeline(ws, pipeline, build_id):
    published_pipeline = pipeline.publish(
        name=PIPELINE_NAME,
        description=build_id,
        version=build_id)

    try:
        pipeline_endpoint = PipelineEndpoint.get(
            workspace=ws,
            name=PIPELINE_ENDPOINT_NAME)
        print("pipeline endpoint exists, add a version")
        pipeline_endpoint.add_default(published_pipeline)
    except Exception:
        print("publish a new pipeline endpoint")
        pipeline_endpoint = PipelineEndpoint.publish(
            workspace=ws,
            name=PIPELINE_ENDPOINT_NAME,
            pipeline=published_pipeline,
            description='NYCtaxi_automl_training_pipeline_endpoint')

    print(f'Published pipeline: {published_pipeline.name}')
    print(f' version: {published_pipeline.version}')
    print(f'Pipeline endpoint: {pipeline_endpoint.name}')
    print('##vso[task.setvariable variable=PIPELINE_ENDPOINT_NAME;]{}'
          .format(pipeline_endpoint.name))
    print('##vso[task.setvariable variable=PIPELINE_ENDPOINT_DEFAULT_VER;]{}'
          .format(pipeline_endpoint.default_version))
    print('##vso[task.setvariable variable=PUBLISHED_PIPELINE_VERSION;]{}'
          .format(published_pipeline.version))
    return pipeline_endpoint


def build_automl_config(is_local_training,
                        user_automl_settings,
                        training_dataset,
                        compute_target):
    featurization_config = FeaturizationConfig()
    featurization_config.add_column_purpose('passengerCount', 'Numeric')

    fixed_automl_settings = {
        'task': 'regression',
        'label_column_name': 'duration',
        'verbosity': logging.INFO,
        'preprocess': False,
        'model_explainability': True,
        'featurization': featurization_config
    }

    if is_local_training:
        automl_config = AutoMLConfig(
            training_data=training_dataset,
            **fixed_automl_settings,
            **user_automl_settings)
    else:
        automl_config = AutoMLConfig(
            path=CODE_PATH,
            training_data=training_dataset,
            compute_target=compute_target,
            **fixed_automl_settings,
            **user_automl_settings)

    return automl_config


if __name__ == '__main__':
    main()
