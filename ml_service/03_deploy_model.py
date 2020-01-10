import os
import sys
import argparse
from azureml.core import Experiment, Run
from azureml.core.model import InferenceConfig
import aml_utils as amlutils

CODE_PATH = '../code'
sys.path.append(os.path.join(os.getcwd(), CODE_PATH))
import consts  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_id',
        type=str,
        dest='run_id',
        help='AML model training run id')
    parser.add_argument(
        '--train_data_folder',
        type=str,
        default='train',
        dest='train_data_folder',
        help='folder on datastore containing training data')

    args = parser.parse_args()
    ws, ds, _, compute_target = amlutils.setup_azureml()
    experiment = Experiment(ws, consts.experiment_name)
    run = Run(experiment, run_id=args.run_id)
    model = amlutils.register_model(
        run, ds, args.train_data_folder, consts.model_name)
    inference_config = create_inference_config(ws)
    uri, token = amlutils.deploy_service(
        ws, model, inference_config,
        consts.service_name, compute_target)
    print('##vso[task.setvariable variable=SVC_URI;]{}'.format(uri))
    print('##vso[task.setvariable variable=SVC_TOKEN;]{}'.format(token))


def create_inference_config(ws):
    inference_env_file = os.path.join(CODE_PATH, 'inference_env.yml')
    inference_env = amlutils.create_azureml_env(
        ws, consts.inference_environment_name, inference_env_file)
    inference_config = InferenceConfig(
        source_directory=CODE_PATH,
        entry_script='score.py',
        environment=inference_env)
    return inference_config


if __name__ == '__main__':
    main()
