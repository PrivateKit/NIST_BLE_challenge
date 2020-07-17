import argparse
from sklearn.model_selection import train_test_split
import os
import random
import math
import time
import torch
import torch.nn as nn

NUM_READINGS_PER_INTERVAL = 150
INTERVAL_LENGTH = 4  # length in seconds of each interval


def load_data(key_path, data_path):
    # first pass to find the various values for the categorical features to use for one-hot encoding
    fixed_variables_key = [set() for _ in range(9)]
    for file_id in os.listdir(data_path):
        if file_id.startswith("."):
            # there are some weird extra files starting with .
            continue
        with open(os.path.join(data_path, file_id), 'r', errors="ignore") as data_file:
            for index in range(7):
                value = data_file.readline().strip().split(",")[1]
                fixed_variables_key[index].add(value)
    with open(key_path, 'r', errors="ignore") as key_file:
        key_file.readline()  # skip header
        for line in key_file:
            transmitter_position, receiver_position = line.split("\t")[1].split("_")
            fixed_variables_key[7].add(transmitter_position)
            fixed_variables_key[8].add(receiver_position)
    fixed_variables_key = [list(x) for x in fixed_variables_key]

    X = list()
    y = list()
    with open(key_path, 'r') as key_file:
        key_file.readline()  # skip header
        for line in key_file:
            record = line.split("\t")
            file_id = record[0]
            file_data = list()
            with open(os.path.join(data_path, file_id)) as data_file:
                # fixed variables tx_device, tx_power, rx_device, tx_carry, rx_carry, rx_pose, tx_pose,
                #                               transmitter_position, receiver_position
                fixed_variables = [data_file.readline().strip().split(",")[1] for _ in range(7)]
                fixed_variables.extend(record[1].split("_"))
                fixed_part = list()
                for variable, key in zip(fixed_variables, fixed_variables_key):
                    fixed_part.extend([int(possible_value == variable) for possible_value in key])
                if len(record) == 5:
                    # this means this file has labels
                    distance = float(record[2])
                    y.append(distance)
                print("Loading file {} with fixed variables of {}".format(file_id, fixed_part))
                interval_start_time = 0
                interval_data = list()
                reading_count = 0
                previous_value = {
                    "Bluetooth": (0,),
                    "Accelerometer": (0,0,0),
                    "Gyroscope": (0,0,0),
                    "Altitude": (0,0,0),
                    "Attitude": (0,0,0),
                    "Gravity": (0,0,0),
                    "Magnetic-field": (0,0,0),
                    "Heading": (0,0,0,0)
                }
                for line in data_file:
                    reading = line.strip().split(",")
                    curr_time = float(reading[0])
                    if (curr_time - interval_start_time) > INTERVAL_LENGTH:
                        if reading_count > NUM_READINGS_PER_INTERVAL:
                            # randomly remove readings
                            for i in range(reading_count - 150):
                                interval_data.pop(math.floor(random.random() * len(interval_data)))
                        else:
                            # randomly duplicate readings
                            # todo: try other methods such as averaging
                            for i in range(NUM_READINGS_PER_INTERVAL - reading_count):
                                interval_data.append(interval_data[math.floor(random.random() * len(interval_data))])
                        file_data.append(interval_data)

                        # reset values
                        interval_start_time = curr_time
                        reading_count = 0
                        interval_data = list()
                        previous_value = {
                            "Bluetooth": (0,),
                            "Accelerometer": (0,0,0),
                            "Gyroscope": (0,0,0),
                            "Altitude": (0,0,0),
                            "Attitude": (0,0,0),
                            "Gravity": (0,0,0),
                            "Magnetic-field": (0,0,0),
                            "Heading": (0,0,0,0)
                        }
                    type = reading[1]
                    if type == "Bluetooth":
                        previous_value[type] = (float(reading[2]), )
                    elif type == "Heading":
                        previous_value[type] = (float(reading[2]), float(reading[3]), float(reading[4]),
                                                float(reading[5]))
                    elif type == "Altitude":
                        previous_value[type] = (float(reading[2]), float(reading[3]))
                    elif type == "Activity":
                        previous_value[type] = (float(reading[2]), reading[3], float(reading[4]))
                    else:
                        previous_value[type] = (float(reading[2]), float(reading[3]), float(reading[4]))
                    # combine the values in previous_value into one giant list
                    interval_data.append([reading for value in previous_value.values() for reading in value] + fixed_part)
                    reading_count += 1
                # the last interval needs to be added manually
                file_data.append(interval_data)
            X.append(file_data)
    return X, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--seed', '-s', type=int, default=100, required=False)

    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    seed = args.seed

    random.seed(seed)

    dev_key_path = os.path.join(data_dir, "docs", "tc4tl_dev_key.tsv")
    test_key_path = os.path.join(data_dir, "docs", "tc4tl_test_metadata.tsv")
    dev_data_path = os.path.join(data_dir, "data", "dev")
    test_data_path = os.path.join(data_dir, "data", "test")

    dev_X, dev_y = load_data(dev_key_path, dev_data_path)
    train_X, val_X, train_y, val_y = train_test_split(dev_X, dev_y)

    # todo: model
    rnn = RNN(150, 128, 1)
    criterion = nn.NLLLoss()

    # train model
    n_iters = len(train_y)
    print_every = 100
    plot_every = 25

    current_loss = 0
    all_losses = []

    for i in range(1, n_iters + 1):
        distance = train_y[i]

