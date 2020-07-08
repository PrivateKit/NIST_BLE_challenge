import matplotlib.pyplot as plt
import argparse
import os
import re
import pandas as pd
import statistics


def find_possible_attenuation_values(input_path):
    """
    iterates through files in directory and finds all the attenuation values
    :param input_path:
    :return: list of values
    """
    attenuation_values = list()
    for file_name in os.listdir(input_path):
        file_path = os.path.join(input_path, file_name)
        if not os.path.isfile(file_path):
            # ignore folders
            continue
        with open(os.path.join(input_path, file_path), "r", encoding="utf8", errors="surrogateescape") as f:
            attenuation_values.extend(map(int, re.findall(r"attenuationValue: (\d+)", f.read())))
    return attenuation_values


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plots pairwise attenuation values on a bar and line graph')
    parser.add_argument('--data-dir', '-d', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, required=True)

    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir)

    # filter out any folder that isn't in the format of "transmitter_to_receiver"
    pairs = [possible_pair for possible_pair in os.listdir(data_dir) if possible_pair.count("_") == 2]

    print("Found {} valid pairs in {}!".format(len(pairs), data_dir))

    column_names = list()  # needed to create the pandas data frame. 
    receiver_column_names = list()  # these are the columns that will contain data to be plotted
    std_column_names = list()  # these are the columns that will contain data for the error bars
    # first pass to get the columns of the data frame
    for pair in pairs:
        transmitter, receiver = pair.split("_to_")
        if receiver not in column_names:
            column_names.append(receiver)
            receiver_column_names.append(receiver)
            std_column_name = receiver + "_std"
            column_names.append(std_column_name)
            std_column_names.append(std_column_name)

    pairwise_attenuation_data = dict()  # transmitter -> [(receiver, mean, std)]
    num_receivers = len(column_names)
    # second pass to fill pairwise_attenuation_data
    print("Calculating mean and standard deviation of the possible attenuation values of each pair... ", end="")
    for pair in pairs:
        possible_attenuation_values = find_possible_attenuation_values(os.path.join(data_dir, pair))
        transmitter, receiver = pair.split("_to_")
        if transmitter not in pairwise_attenuation_data:
            pairwise_attenuation_data[transmitter] = [None] * num_receivers
        receiver_index = column_names.index(receiver)
        mean = statistics.mean(possible_attenuation_values)
        std = statistics.stdev(possible_attenuation_values)

        pairwise_attenuation_data[transmitter][receiver_index] = mean
        pairwise_attenuation_data[transmitter][receiver_index + 1] = std

    pairwise_attenuation_df = pd.DataFrame.from_dict(pairwise_attenuation_data, orient='index', columns=column_names)
    print("done!")

    # bar graph
    print("Plotting bar graph...", end="")
    pairwise_attenuation_df[receiver_column_names].plot(kind="bar",
                                                        yerr=pairwise_attenuation_df[std_column_names].values.T,
                                                        alpha=0.5,
                                                        error_kw=dict(ecolor='k'))
    plt.savefig(os.path.join(output_dir, "bar_graph.png"))
    print(" done!")

    # line graph
    print("Plotting line graph...", end="")
    pairwise_attenuation_df[receiver_column_names].plot(kind="line",
                                                        yerr=pairwise_attenuation_df[std_column_names].values.T,
                                                        alpha=0.5,
                                                        fmt="-o")
    plt.savefig(os.path.join(output_dir, "line_graph.png"))
    print(" done!")