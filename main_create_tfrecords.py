from buffer import create_tfrecords


def main():
    original_dataset_dir = 'Breakout/1/replay_logs'
    dataset_dir = 'tfrecords_dqn_replay_dataset/'

    create_tfrecords(original_dataset_dir=original_dataset_dir,
                     dataset_dir=dataset_dir,
                     num_data_files=50,  # default=50
                     use_samples_per_file=10000,  # default=10000
                     num_chunks=10  # default=10
                     )


if __name__ == '__main__':
    main()
