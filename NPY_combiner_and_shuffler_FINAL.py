import numpy as np
import os

# Set the path to the directory containing the .npy files
data_dir = '/home/20ucc067/Shubhang/FINAL_NPY'

# Create an empty list to store file paths
file_paths = []

# Collect the file paths for all .npy files in the directory
for i in range(1, 12):
    file_path = os.path.join(data_dir, f"{i}.npy")
    file_paths.append(file_path)

# Create a memory-mapped array for the combined data
combined_data = np.memmap('X.npy', dtype='uint8', mode='w+', shape=(27447, 128, 128))

# Assign labels as before
labels = []
count=0
# Loop through file paths and load data
for i, file_path in enumerate(file_paths):
    file_data = np.load(file_path)
    label = i + 1
    combined_data[count:count+file_data.shape[0]] = file_data
    count+=file_data.shape[0]
    labels.extend([label] * file_data.shape[0])
combined_data.flush()
# Save the labels
combined_labels = np.array(labels)
np.save('Y.npy', combined_labels)
combined_data=np.array(combined_data)
np.save('X2.npy', combined_data)
#Y2==Y
shuffle_indices = np.arange(len(combined_data))
np.random.shuffle(shuffle_indices)
# Shuffle the combined data and labels in the same order
combined_data = combined_data[shuffle_indices]
combined_labels = combined_labels[shuffle_indices]
combined_data=np.array(combined_data)
np.save('X3.npy', combined_data)
np.save('Y3.npy', combined_labels)
