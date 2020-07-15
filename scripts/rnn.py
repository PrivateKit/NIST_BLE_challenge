from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# TODO: Make argparse


def compute_averages(bluetooth_data, accelerometer_data, gyroscope_data, attitude_data, gravity_data,
                     magnetic_field_data, file_data):
    if len(bluetooth_data) == 0:
        avg_bluetooth_data = file_data[-1][1] if len(file_data) > 0 else 0  # todo: find proper value for this case
    else:
        avg_bluetooth_data = sum(bluetooth_data) / len(bluetooth_data)
    if len(accelerometer_data) == 0:
        avg_accelerometer_data = file_data[-1][1] if len(file_data) > 0 else (0, 0, 0)  # todo: find proper value for this case
    else:
        avg_accelerometer_data = [sum(col) / len(col) for col in zip(*accelerometer_data)]
    if len(gyroscope_data) == 0:
        avg_gyroscope_data = file_data[-1][2] if len(file_data) > 0 else (0, 0, 0)  # todo: find proper value for this case
    else:
        avg_gyroscope_data = [sum(col) / len(col) for col in zip(*gyroscope_data)]
    if len(attitude_data) == 0:
        avg_attitude_data = file_data[-1][3] if len(file_data) > 0 else (0, 0, 0)  # todo: find proper value for this case
    else:
        avg_attitude_data = [sum(col) / len(col) for col in zip(*attitude_data)]
    if len(gravity_data) == 0:
        avg_gravity_data = file_data[-1][4] if len(file_data) > 0 else (0, 0, 0)  # todo: find proper value for this case
    else:
        avg_gravity_data = [sum(col) / len(col) for col in zip(*gravity_data)]
    if len(magnetic_field_data) == 0:
        avg_magnetic_field_data = file_data[-1][5] if len(file_data) > 0 else (0, 0, 0)  # todo: find proper value for this case
    else:
        avg_magnetic_field_data = [sum(col) / len(col) for col in zip(*magnetic_field_data)]

    return np.array([avg_bluetooth_data,] + avg_accelerometer_data + avg_gyroscope_data + avg_attitude_data
                    + avg_gravity_data + avg_magnetic_field_data)


def find_shortest_step_size(key_paths):
    step_sizes = set()
    for key_path in key_paths:
        with open(key_path, 'r') as key_file:
            is_labels = len(key_file.readline().split("\t")) == 4
            step_size_index = 2 if is_labels else 3
            for line in key_file:
                step_size = line.split("\t")[step_size_index].strip()
                if step_size == "inf":
                    continue
                step_size = int(step_size)
                step_sizes.add(step_size)
    return min(step_sizes)


def load_data(key_path, data_path, step_size):
    X = list()
    y = list()
    with open(key_path, 'r') as key_file:
        key_file.readline()
        for line in key_file:
            record = line.split("\t")
            file_id = record[0]
            file_data = list()
            with open(os.path.join(data_path, file_id)) as data_file:
                # fixed variables
                tx_device, tx_power, rx_device, tx_carry, rx_carry, rx_pose, tx_pose = \
                    [data_file.readline().strip().split(",")[1] for _ in range(7)]
                transmitter_position, receiver_position = record[1].split("_")
                fixed_part = [tx_device, tx_power, rx_device, tx_carry, rx_carry, rx_pose, tx_pose,
                                   transmitter_position, receiver_position]
                if len(record) == 5:
                    # has labels
                    distance = float(record[2])
                    y.append(distance)
                # fixed_part = np.array(fixed_part)
                # todo: add categorical variables with one hot encoding
                fixed_part = np.array([])
                print("Loading file {} with fixed variables of {}".format(file_id, fixed_part))

                curr_time = 0
                bluetooth_data = list()
                accelerometer_data = list()
                gyroscope_data = list()
                attitude_data = list()
                gravity_data = list()
                magnetic_field_data = list()
                for line in data_file:
                    reading = line.strip().split(",")
                    time = float(reading[0])
                    if (time - curr_time) > step_size:
                        variable_part = compute_averages(bluetooth_data, accelerometer_data, gyroscope_data,
                                                         attitude_data, gravity_data, magnetic_field_data, file_data)
                        file_data.append(np.concatenate([variable_part, fixed_part]))

                        curr_time = time
                        bluetooth_data = list()
                        accelerometer_data = list()
                        gyroscope_data = list()
                        attitude_data = list()
                        gravity_data = list()
                        magnetic_field_data = list()
                    type = reading[1]
                    if type == "Bluetooth":
                        bluetooth_data.append(float(reading[2]))
                    elif type == "Accelerometer":
                        accelerometer_data.append((float(reading[2]), float(reading[3]), float(reading[4])))
                    elif type == "Gyroscope":
                        gyroscope_data.append((float(reading[2]), float(reading[3]), float(reading[4])))
                    elif type == "Attitude":
                        attitude_data.append((float(reading[2]), float(reading[3]), float(reading[4])))
                    elif type == "Gravity":
                        gravity_data.append((float(reading[2]), float(reading[3]), float(reading[4])))
                    elif type == "Magnetic-field":
                        magnetic_field_data.append((float(reading[2]), float(reading[3]), float(reading[4])))

                # Add last interval
                variable_part = compute_averages(bluetooth_data, accelerometer_data, gyroscope_data,
                                                 attitude_data, gravity_data, magnetic_field_data, file_data)
                file_data.append(np.concatenate([variable_part, fixed_part]))
            X.append(np.array(file_data))
    return X, y


data_dir = "/home/sheshank/tc4tl"

dev_key_path = os.path.join(data_dir, "docs", "tc4tl_dev_key.tsv")
test_key_path = os.path.join(data_dir, "docs", "tc4tl_test_metadata.tsv")
dev_data_path = os.path.join(data_dir, "data", "dev")
test_data_path = os.path.join(data_dir, "data", "test")

step_size = find_shortest_step_size({dev_key_path, test_key_path})
print("Using step size of {}".format(step_size))

dev_X, dev_y = load_data(dev_key_path, dev_data_path, step_size)
train_X, val_X, train_y, val_y = train_test_split(dev_X, dev_y)

# test_X, _ = load_data(test_key_path, test_data_path, step_size)

model = keras.Sequential()

# Add a LSTM layer with 128 internal units.
model.add(LSTM(128, input_shape=(None, 16)))

model.add(BatchNormalization())

# Add a Dense layer with 10 units.
model.add(Dense(1))

model.summary()

model.compile(
    optimizer="adam",
    loss="categorical_cross_entropy",
    metrics=["accuracy"],
)

model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=124, epochs=1)
