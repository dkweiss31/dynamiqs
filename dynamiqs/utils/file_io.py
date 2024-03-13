import os
import glob
import re
import h5py


def generate_file_path(extension, file_name, path):
    # Ensure the path exists.
    os.makedirs(path, exist_ok=True)

    # Create a save file name based on the one given; ensure it will
    # not conflict with others in the directory.
    max_numeric_prefix = -1
    for file_name_ in glob.glob(os.path.join(path, "*")):
        if f"_{file_name}.{extension}" in file_name_:
            numeric_prefix = int(
                re.match(r"(\d+)_", os.path.basename(file_name_)).group(1)
            )
            max_numeric_prefix = max(numeric_prefix, max_numeric_prefix)

    # Generate the file path.
    file_path = os.path.join(
        path, f"{str(max_numeric_prefix + 1).zfill(5)}_{file_name}.{extension}"
    )
    return file_path


def extract_info_from_h5(filepath):
    data_dict = {}
    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            data_dict[key] = f[key][()]
        param_dict = dict(f.attrs.items())
    return data_dict, param_dict


def append_to_h5(filepath, data_dict):
    with h5py.File(filepath, "a") as f:
        for key, val in data_dict.items():
            f[key].resize(f[key].shape[0] + 1, axis=0)
            f[key][-1] = val


def write_to_h5_multi(filepath, data_dict, param_dict):
    with h5py.File(filepath, "a") as f:
        for key, val in data_dict.items():
            f.create_dataset(key, data=[val], chunks=True, maxshape=(None, *val.shape))
        for kwarg in param_dict.keys():
            try:
                f.attrs[kwarg] = param_dict[kwarg]
            except TypeError:
                f.attrs[kwarg] = str(param_dict[kwarg])