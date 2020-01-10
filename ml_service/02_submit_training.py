import os
import sys
import argparse
from azureml.core import Experiment
from azureml.train.estimator import Estimator
import aml_utils as amlutils

CODE_PATH = '../code'
sys.path.append(os.path.join(os.getcwd(), CODE_PATH))
import consts  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        type=str,
        default='input',
        dest='input_folder',
        help='path in the datastore where training data is located')
    parser.add_argument(
        '--build_id',
        type=str,
        default='manual',
        dest='build_id',
        help='build id from ci/cd pipelines')

    args = parser.parse_args()
    ws, ds, compute_target, _ = amlutils.setup_azureml()
    experiment = Experiment(ws, consts.experiment_name)
    est = create_estimator(ws, ds, compute_target, args.input_folder)
    run = experiment.submit(est, tags={'build_id': args.build_id})
    run.wait_for_completion(show_output=True)
    print('##vso[task.setvariable variable=AML_RUN_ID;]{}'.format(run.id))
    """
    runid = 'nyctaxi_experiment_1578534499_a07d83ac'
    print('##vso[task.setvariable variable=AML_RUN_ID;]{}'.format(runid))
    """


def create_estimator(ws, ds, compute_target, input_folder):
    training_env_file = os.path.join(CODE_PATH, 'train_env.yml')
    training_env = amlutils.create_azureml_env(
        ws, consts.train_environment_name, training_env_file)

    script_params = {
        '--data_folder': ds.path(input_folder).as_mount()
    }
    est = Estimator(
            source_directory=CODE_PATH,
            entry_script='train.py',
            script_params=script_params,
            compute_target=compute_target,
            environment_definition=training_env)
    return est


if __name__ == '__main__':
    main()
