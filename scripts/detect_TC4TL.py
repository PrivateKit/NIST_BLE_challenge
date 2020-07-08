import argparse
import os
import random
import math
import pandas as pd
import re
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


def load_data(data_dir):
    data = list()
    for folder_name in [name for name in os.listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, name))]:
        try:
            transmitter_location, receiver_location = folder_name.split("_")
        except ValueError:
            print("Skipping folder: {}".format(folder_name))
            continue
        file_names = os.listdir(os.path.join(data_dir, folder_name))
        for file_name in file_names:
            try:
                distance = int(re.findall(r"_(\d+)ft_l", file_name)[0])
            except IndexError:
                # ignore files that aren't log files
                print("Skipping file: {}".format(file_name))
                continue
            with open(os.path.join(data_dir, folder_name, file_name), 'r') as f:
                for i in range(10):
                    # ignoring first 10 lines, only using bluetooth data
                    f.readline()
                # As per README.md, only using BlueProxTx readings
                bluetooth_data = [line.split(",") for line in f.read().split("\n") if "BlueProxTx" in line]
                for record in bluetooth_data:
                    # remove data that is not needed
                    record.pop(1)
                    record.pop(1)
                    record.pop(2)
                    record.pop(2)
                    # remove decimal as datetime crashes with decimal in seconds
                    # some have "T" instead of a space in between days and hours
                    record[0] = record[0][:record[0].index(".")].replace("T", " ")
                # take random pairs of readings and take the time in between
                random.shuffle(bluetooth_data)
                for i in range(math.floor(len(bluetooth_data)/2)):
                    record_1 = bluetooth_data[i]
                    record_2 = bluetooth_data[i+1]
                    try:
                        # there are some entries with messed up data, just skip for now
                        time_1 = datetime.strptime(record_1[0], "%Y-%m-%d %H:%M:%S")
                        time_2 = datetime.strptime(record_2[0], "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        print("Skipping badly formatted time: {}".format(time_1))
                        continue
                    time_interval = abs(time_1 - time_2).seconds
                    # todo: add phone types as a feature
                    data.append((float(record_1[1]), float(record_2[1]), transmitter_location, receiver_location,
                                 time_interval, distance))

    df = pd.DataFrame(data)
    df.columns = ["rssi1", "rssi2", "transmitter_position", "receiver_position", "time", "distance"]
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='detects TC4TL')
    parser.add_argument('--data-dir', '-d', type=str, required=True, help='path to MIT-Matrix-Data repo')
    parser.add_argument('--model', '-m', type=str, required=True, help='xgboost or random-forest')
    parser.add_argument('--seed', '-s', type=int, default=100, required=False)

    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    model_name = args.model
    seed = args.seed

    random.seed(seed)

    print("Using {} model on {}".format(model_name, data_dir))

    df = load_data(data_dir)

    # Create X
    features = ['rssi1', 'rssi2', 'transmitter_position', 'receiver_position']
    categorical_features = ['transmitter_position', 'receiver_position']
    X = df[features]

    # One hot encoding for categorical columns
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[categorical_features]))

    # put index back
    OH_cols.index = X.index

    # Remove old categorical columns
    num_X = X.drop(categorical_features, axis=1)

    # Add the one-hot encoded columns
    OH_X = pd.concat([num_X, OH_cols], axis=1)

    ##### Predicting Distance
    y_distance = df.distance
    train_X_distance, val_X_distance, train_y_distance, val_y_distance = train_test_split(OH_X, y_distance, random_state=1)
    if model_name == "xgboost":
        distance_model = XGBRegressor(random_state=seed)
    elif model_name == "random-forest":
        distance_model = RandomForestRegressor(random_state=seed)
    else:
        raise Exception("Model {} not supported".format(model_name))
    distance_model.fit(train_X_distance, train_y_distance)
    distance_mae = mean_absolute_error(distance_model.predict(val_X_distance), val_y_distance)
    print("[{}] Mean Absolute Error when predicting distance: {}".format(model_name, distance_mae))

    ##### Predicting Time
    y_time = df.time
    train_X_time, val_X_time, train_y_time, val_y_time = train_test_split(OH_X, y_time, random_state=1)
    if model_name == "xgboost":
        time_model = XGBRegressor(random_state=seed)
    elif model_name == "random-forest":
        time_model = RandomForestRegressor(random_state=seed)
    else:
        raise Exception("Model {} not supported".format(model_name))
    time_model.fit(train_X_time, train_y_time)
    time_mae = mean_absolute_error(time_model.predict(val_X_time), val_y_time)
    print("[{}] Mean Absolute Error when predicting distance: {}".format(model_name, time_mae))
