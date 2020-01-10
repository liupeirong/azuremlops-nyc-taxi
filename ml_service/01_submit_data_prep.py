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
        help='path in the datastore where raw data is located')
    parser.add_argument(
        '--output_folder',
        type=str,
        default='train',
        dest='output_folder',
        help='path in the datastore where to output processed data')
    parser.add_argument(
        '--build_id',
        type=str,
        default='manual',
        dest='build_id',
        help='build id from ci/cd pipeline')

    args = parser.parse_args()
    ws, ds, compute_target, _ = amlutils.setup_azureml()
    training_env_file = os.path.join(CODE_PATH, 'train_env.yml')
    training_env = amlutils.create_azureml_env(
        ws, consts.train_environment_name, training_env_file)
    experiment = Experiment(ws, consts.data_experiment_name)
    est = create_estimator(
        ws, ds, compute_target, training_env,
        args.input_folder, args.output_folder)
    run = experiment.submit(est, tags={'build_id': args.build_id})
    run.wait_for_completion(show_output=True)


def create_estimator(ws, ds, compute_target, run_env,
                     input_folder, output_folder):
    script_params = {
        '--raw_folder': ds.path(input_folder).as_mount(),
        '--processed_folder': ds.path(output_folder).as_mount()
    }
    est = Estimator(
            source_directory=CODE_PATH,
            entry_script='data_prep.py',
            script_params=script_params,
            compute_target=compute_target,
            environment_definition=run_env)
    return est


if __name__ == '__main__':
    main()
