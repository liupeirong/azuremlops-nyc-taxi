import time
import argparse
from azureml.core import Run
import utils


def generate_filename():
    timestr = time.strftime("%Y%m%d-%H%M%S")
    return timestr + '.csv'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--raw_folder',
        type=str,
        default='raw',
        dest='raw_folder',
        help='folder in datastore where raw data is located')
    parser.add_argument(
        '--processed_folder',
        type=str,
        default='train',
        dest='processed_folder',
        help='folder in datastore where to output processed data')
    args = parser.parse_args()

    run = Run.get_context()
    filename = generate_filename()

    run.tag('raw_folder',
            utils.last_two_folders_if_exists(args.raw_folder))
    run.tag('processed_folder',
            utils.last_two_folders_if_exists(args.processed_folder))
    run.tag('output_file', filename)

    df = utils.read_raw_data(args.raw_folder)
    df = utils.process_raw_data(df)
    utils.write_train_data(df, args.processed_folder, filename)

    run.log_list("shape", [df.shape[0], df.shape[1]])


if __name__ == '__main__':
    main()
