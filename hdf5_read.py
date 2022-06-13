import h5py
import numpy as np
filename = "reconstructed_poses_noratap.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[1]

    # Get the data
    data = list(f[a_group_key])
    pose_data = data[0].reshape((24, 3))

    for i in range(0, 24):
        print(f"{i}:{np.degrees(pose_data[i])}")

    print(len(data[0]))
