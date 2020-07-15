import matplotlib.pyplot as plt
import statistics
import os

data_dir = "/home/sheshank/tc4tl"

key_path = os.path.join(data_dir, "docs", "tc4tl_dev_key.tsv")
data_path = os.path.join(data_dir, "data", "dev")
step_size_to_file_id = dict()
with open(key_path, 'r') as key_file:
    key_file.readline()
    for line in key_file:
        file_id, _, _, step_size, _ = line.strip().split("\t")
        if step_size == "inf":
            continue
        step_size = int(step_size)
        if step_size not in step_size_to_file_id:
            step_size_to_file_id[step_size] = set()
        step_size_to_file_id[step_size].add(file_id)
step_sizes = list(step_size_to_file_id.keys())
step_sizes.sort()
means = {
    "Bluetooth": list(),
    "Accelerometer": list(),
    "Gyroscope": list(),
    "Altitude": list(),
    "Activity": list(),
    "Attitude": list(),
    "Gravity": list(),
    "Magnetic-field": list(),
    "Heading": list()
}
stds = {
    "Bluetooth": list(),
    "Accelerometer": list(),
    "Gyroscope": list(),
    "Altitude": list(),
    "Activity": list(),
    "Attitude": list(),
    "Gravity": list(),
    "Magnetic-field": list(),
    "Heading": list()
}
for step_size in step_sizes:
    step_size_type_counts = {
        "Bluetooth": list(),
        "Accelerometer": list(),
        "Gyroscope": list(),
        "Altitude": list(),
        "Attitude": list(),
        "Activity": list(),
        "Gravity": list(),
        "Magnetic-field": list(),
        "Heading": list()
    }
    for file_id in step_size_to_file_id[step_size]:
        last_interval_end = 0
        file_type_counts = {
            "Bluetooth": 0,
            "Accelerometer": 0,
            "Gyroscope": 0,
            "Altitude": 0,
            "Activity": 0,
            "Attitude": 0,
            "Gravity": 0,
            "Magnetic-field": 0,
            "Heading": 0
        }
        with open(os.path.join(data_path, file_id), 'r') as f:
            for i in range(7):
                f.readline()
            for line in f:
                record = line.split(",")
                time = float(record[0])
                if (time - last_interval_end) > 4:
                    last_interval_end = time
                    for count_type in file_type_counts.keys():
                        step_size_type_counts[count_type].append(file_type_counts[count_type])
                    file_type_counts = {
                        "Bluetooth": 0,
                        "Accelerometer": 0,
                        "Gyroscope": 0,
                        "Altitude": 0,
                        "Activity": 0,
                        "Attitude": 0,
                        "Gravity": 0,
                        "Magnetic-field": 0,
                        "Heading": 0
                    }
                type = record[1]
                if type not in file_type_counts:
                    continue
                file_type_counts[type] += 1
    for count_type in step_size_type_counts.keys():
        means[count_type].append(statistics.mean(step_size_type_counts[count_type]))
        stds[count_type].append(statistics.stdev(step_size_type_counts[count_type]))
print("Step Sizes: {}".format(step_sizes))
print("Means: {}".format(means))
print("Stds: {}".format(stds))
plt.errorbar(list(step_sizes), means['Bluetooth'], stds['Bluetooth'], elinewidth=5, ecolor='red', color='red')
plt.errorbar(list(step_sizes), means['Accelerometer'], stds['Accelerometer'], elinewidth=5, ecolor='red', color='green')
plt.errorbar(list(step_sizes), means['Gyroscope'], stds['Gyroscope'], elinewidth=5, ecolor='red', color='blue')
plt.errorbar(list(step_sizes), means['Altitude'], stds['Altitude'], elinewidth=5, ecolor='red', color='orange')
plt.errorbar(list(step_sizes), means['Activity'], stds['Activity'], elinewidth=5, ecolor='red', color='grey')
plt.errorbar(list(step_sizes), means['Attitude'], stds['Attitude'], elinewidth=5, ecolor='red', color='purple')
plt.errorbar(list(step_sizes), means['Magnetic-field'], stds['Magnetic-field'], elinewidth=5, ecolor='red', color='pink')
plt.errorbar(list(step_sizes), means['Heading'], stds['Heading'], elinewidth=5, ecolor='red', color='yellow')
plt.xlabel("Step Size (s)")
plt.ylabel("Count")
plt.show()