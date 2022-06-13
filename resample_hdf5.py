import h5py
filename = "masks"

with h5py.File(f"{filename}.hdf5", "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])

    # Resample 25 fps to 10 fps
    # 5 = (5/25)*25
    resample_result = []
    for i in range(0, len(data), 5):
        resample_result.append(data[i])

    print(f"Resample Size:{len(resample_result)}")

    # Write data to HDF5
    with h5py.File(f"{filename}_resample.hdf5", "w") as data_file:
        data_file.create_dataset(a_group_key, data=resample_result)
