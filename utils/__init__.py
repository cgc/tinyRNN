from .logger import *
from .goto_root_dir import *
from contextlib import contextmanager
import pathlib
import numpy as np
def get_current_file_name(file_string):
    """Automatically generate the current file name

    Args:
        file_string: __file__

    Returns:
        the file name without .py
    """
    return os.path.basename(file_string)[:-3] # remove .py

@contextmanager
def set_os_path_auto():
    # detect current os
    if sys.platform == 'win32':
        posix_backup = pathlib.PosixPath
        try:
            pathlib.PosixPath = pathlib.WindowsPath
            yield
        finally:
            pathlib.PosixPath = posix_backup
    elif sys.platform in ('linux', 'darwin'):
        posix_backup = pathlib.WindowsPath
        try:
            pathlib.WindowsPath = pathlib.PosixPath
            yield
        finally:
            pathlib.WindowsPath = posix_backup
    else:
        raise NotImplementedError


def pd_full_print_context():
    """Print the full dataframe in the console."""
    import pandas as pd
    return pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False, 'display.max_colwidth', None)


def save_config(config, save_path, save_name):
    """Save config to disk."""
    import joblib
    from copy import deepcopy
    import json
    save_path = Path(save_path)
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(config, save_path / (save_name + '.pkl'))

    config_json = deepcopy(config)
    try:
        for k in config_json.keys():
            if 'index' in k: # convert np array to list, for the keys of "index"; better generalize this
                if not isinstance(config_json[k], list):
                    config_json[k] = config_json[k].tolist()
            if 'path' in k:
                config_json[k] = str(config_json[k])
            if isinstance(config_json[k], np.int32) or isinstance(config_json[k], np.int64):
                config_json[k] = int(config_json[k])
        with open(save_path / (save_name + '.json'), 'w') as f:
            json.dump(config_json, f, indent=4)
    except:
        print('config saving failed')
        for k in config_json.keys():
            print(k, type(config_json[k]))
        raise ValueError('config saving failed')


def highlighted_print(df, by_key='sub_count'):
    # ANSI escape codes for red text and reset
    RED = "\033[91m"
    RESET = "\033[0m"

    df_str = df.to_string(index=False).split('\n')
    headers = df_str[0]
    rows = df_str[1:]

    # Apply red color to rows where sub_count is different from the most frequent
    colored_rows = []
    sub_counts = df[by_key].value_counts()
    if len(sub_counts) == 0:
        always_colored = True
    else:
        always_colored = False
        most_freq_sub_count = df[by_key].value_counts().idxmax()
    for row in rows:
        sub_count_index = df.columns.get_loc(by_key)
        row_values = row.split()
        if always_colored or len(row_values) > sub_count_index and row_values[sub_count_index] != str(most_freq_sub_count):
            colored_rows.append(RED + row + RESET)
        else:
            colored_rows.append(row)

    print(headers)
    for row in colored_rows:
        print(row)