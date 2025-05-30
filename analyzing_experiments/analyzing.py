import os

import joblib
import numpy as np
import torch

from utils import set_os_path_auto
from utils import goto_root_dir
import pathlib
from pathlib import Path
from path_settings import *
import pandas as pd
from agents import Agent
from datasets import Dataset
from contextlib import contextmanager
import sys
from utils.logger import PrinterLogger

@contextmanager
def set_posix_windows():
    """Temporarily change the posix path to windows path for loading the models."""
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def pd_full_print_context():
    """Print the full dataframe in the console."""
    return pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False, 'display.max_colwidth', None)


def get_config_from_path(model_path):
    """Get the config from the model path."""
    model_path = Path(model_path)
    if MODEL_SAVE_PATH.name not in model_path.parts:
        model_path = MODEL_SAVE_PATH / model_path
    with set_posix_windows():
        config = joblib.load(model_path / 'config.pkl')
        config['model_path'] = Path(str(config['model_path']))
    return config


def get_model_from_config(config, best_device=True):
    """Get the model from the config."""
    if best_device and 'rnn_type' in config and torch.cuda.is_available():
        config['device'] = 'cuda'
    ag = Agent(config['agent_type'], config=config)
    ag.load(config['model_path'], strict=False) # for the dummy variable
    return ag


def transform_model_format(ob, source='', target='', best_device=True):
    """
    Transform the model format from one format to another.
    row->path; row->config
    path<->config
    config->agent
    Args:
        ob: The model object.
        source: 'row', 'path', 'config'
        target: 'path', 'config', 'agent'

    Returns:
        The transformed model object.
    """
    assert source in ['row', 'path', 'config']
    assert target in ['path', 'config', 'agent']
    if source == 'row':
        ob = ob['config']
        source = 'config'
    # now source is either 'path' or 'config'
    if target == 'path':
        if source == 'path':
            return Path(ob)
        else: # source == 'config'
            return Path(ob['model_path'])
    # now target is either 'config' or 'agent'
    if source == 'path':
        ob = get_config_from_path(ob)
    # now source is 'config'
    if target == 'config':
        return ob
    else: # target is 'agent'
        return get_model_from_config(ob, best_device=best_device)


def get_model_path_in_exp(exp_folder, name_filter=''):
    """Obtain the path of all models in the training experiment folder.

    Args:
        exp_folder (str): The name of the experiment folder.
        name_filter (str, optional): The filter to select the model. Defaults to ''.

    Returns:
        list: The list of paths of the models.
    """
    path = MODEL_SAVE_PATH / Path(exp_folder)
    model_paths = []
    for p in path.rglob("*"): # recursively search all subfolders
        if p.name == 'config.pkl':
            model_paths.append(p.parent)
    if len(model_paths)==0:
        print('No models found at ',path)
        raise ValueError()
    return model_paths

def construct_behav_data_spec(config):
    behav_data_spec = config['behav_data_spec']
    if isinstance(behav_data_spec, list): # assemble the dict for behavior data specification
        behav_data_spec = {k: config[k] for k in behav_data_spec}
    return behav_data_spec


def insert_model_acc_in_df(summary,dataset_loading_every_time=False, include_acc_filter=lambda row: True, ):
    new_summary = []
    def mask_with_trial_index(mask, trial_index):
        if isinstance(mask, tuple):
            mask = mask[0] # ignore the mask[1] of 999 for actions in target
        if trial_index is None:
            return mask

        assert len(mask.shape) == 1 or mask.shape[1] == 2 # either 1 stage per trial or 2 stages per trial
        select_trials = np.zeros_like(mask)
        select_trials[trial_index] = 1
        return (mask * select_trials).flatten() # this make no difference for 1 stage per trial, but for 2 stages per trial, it's necessary to match the total_corrected_trials shape

    behav_dt, augment_ratio = None, None
    for i, row in summary.iterrows():
        if not include_acc_filter(row):
            new_summary.append(row)
            continue
        config = transform_model_format(row, source='row', target='config')
        # ag = transform_model_format(config, source='config', target='agent')
        model_path = transform_model_format(row, source='config', target='path')
        # if we want to select trials, meaning that each block has the same trial num; then there's no padding, and demask is not required
        demask= False if 'train_trial_index' in config else True
        total_scores, behav_dt, augment_ratio = run_scores_for_each_model(row, dataset_loading_every_time=dataset_loading_every_time, include_data='trainvaltest',
                                                                          behav_dt=behav_dt, augment_ratio=augment_ratio, pointwise_loss=True, demask=demask)
        assert 'total_corrected_trials' in total_scores
        total_corrected_trials = total_scores['total_corrected_trials']
        trainvaltest_indexes = np.unique(np.concatenate([config['train_index'], config['val_index'], config['test_index']]))

        if 'mask' in total_scores and total_scores['mask'] is not None:
            mask = total_scores['mask']
        else:
            mask = [np.ones_like(x) for x in total_corrected_trials]
        # find new positions in trainvaltest_indexes
        train_index = np.where(np.isin(trainvaltest_indexes, config['train_index']))[0]
        val_index = np.where(np.isin(trainvaltest_indexes, config['val_index']))[0]
        test_index = np.where(np.isin(trainvaltest_indexes, config['test_index']))[0]
        train_trial_index = config['train_trial_index'] if 'train_trial_index' in config else None
        val_trial_index = config['val_trial_index'] if 'val_trial_index' in config else None
        test_trial_index = config['test_trial_index'] if 'test_trial_index' in config else None
        new_row = row.copy()
        train_trials = np.concatenate([total_corrected_trials[i]*mask_with_trial_index(mask[i], train_trial_index) for i in train_index])
        train_trials_num = np.sum(np.concatenate([mask_with_trial_index(mask[i], train_trial_index) for i in train_index]))
        val_trials = np.concatenate([total_corrected_trials[i]*mask_with_trial_index(mask[i], val_trial_index) for i in val_index])
        val_trials_num = np.sum(np.concatenate([mask_with_trial_index(mask[i], val_trial_index) for i in val_index]))
        new_row['train_acc'] = np.sum(train_trials) / train_trials_num
        new_row['val_acc'] = np.sum(val_trials) / val_trials_num
        new_row['trainval_acc'] = (new_row['train_acc'] * train_trials_num + new_row['val_acc'] * val_trials_num) / (train_trials_num + val_trials_num)
        new_row['test_acc'] = np.sum(np.concatenate([total_corrected_trials[i]*mask_with_trial_index(mask[i], test_trial_index) for i in test_index])) / np.sum(np.concatenate([mask_with_trial_index(mask[i], test_trial_index) for i in test_index]))
        new_summary.append(new_row)
    return pd.DataFrame(new_summary)

def get_model_test_scores(row, behav_dt):
    """Get the model test scores from the row of a dataframe. (Currently not used)

    Args:
        row (pd.Series): The row of the dataframe.
        behav_dt (Dataset): The loaded dataset.

    Returns:
        scores: The scores of the model on the test set.
    """
    config = transform_model_format(row, source='row', target='config')
    ag = transform_model_format(config, source='config', target='agent')
    # behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
    behav_dt = behav_dt.behav_to(config)
    data = behav_dt.get_behav_data(config['test_index'], config)
    test_model_pass = ag.forward(data)
    return test_model_pass['output'].detach().cpu().numpy()



def insert_model_test_scores_in_df(df):
    """Insert the model test scores into the dataframe. (Currently not used)

    Args:
        df (pd.DataFrame): The dataframe containing the model paths.

    Returns:
        pd.DataFrame: The dataframe with the test scores.
    """
    config = df.iloc[0]['config']
    behav_dt = Dataset(config['dataset'], behav_data_spec=config['behav_data_spec'])
    df['test_scores'] = df.apply(lambda x: get_model_test_scores(x, behav_dt), axis=1)
    return df


def combine_test_scores(df, group_by_keys):
    """Combine the test scores from the inner folds. (Currently not used)
    TODO
    """
    df = df.groupby(group_by_keys, as_index=False).agg({'test_scores': lambda x: list(x.mean())})
    df['test_scores'] = df['test_scores'].apply(np.array)

def run_scores_exp(exp_folder,best_for_test=False, model_filter=None,overwrite_config=None, pointwise_loss=False,
                   demask=True, include_data='all', has_rnn=True, has_cog=True, suffix_name='',dataset_loading_every_time=True):
    """pointwise_loss=False will return #times=task_trials+1, and pointwise_loss=True will return #times=task_trials"""
    ana_exp_path = ANA_SAVE_PATH / exp_folder
    print(ana_exp_path)
    if best_for_test:
        with set_os_path_auto():
            if has_rnn:
                rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary_based_on_test.pkl')
            else:
                rnn_summary = pd.DataFrame()
            if has_cog:
                cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary_based_on_test.pkl')
            else:
                cog_summary = pd.DataFrame()
    else:
        with set_os_path_auto():
            if has_rnn:
                rnn_summary = joblib.load(ana_exp_path / 'rnn_final_best_summary.pkl')
            else:
                rnn_summary = pd.DataFrame()
            if has_cog:
                cog_summary = joblib.load(ana_exp_path / 'cog_final_best_summary.pkl')
            else:
                cog_summary = pd.DataFrame()
    if overwrite_config is None:
        overwrite_config = {}
    if model_filter is not None:
        for k,v in model_filter.items():
            if k in rnn_summary.columns:
                rnn_summary = rnn_summary[rnn_summary[k]==v]
                print(f'filter rnn_summary by {k}={v}')
            if k in cog_summary.columns:
                cog_summary = cog_summary[cog_summary[k]==v]
                print(f'filter cog_summary by {k}={v}')
            if k not in rnn_summary.columns and k not in cog_summary.columns:
                raise ValueError(f'{k} not in rnn_summary or cog_summary')
    behav_dt = None
    augment_ratio = None
    for i, row in pd.concat([cog_summary, rnn_summary], axis=0, join='outer').iterrows():
        save_dict, behav_dt, augment_ratio = run_scores_for_each_model(row,suffix_name=suffix_name, overwrite_config=overwrite_config, dataset_loading_every_time=dataset_loading_every_time,
                                              behav_dt=behav_dt, augment_ratio=augment_ratio,include_data=include_data,pointwise_loss=pointwise_loss,demask=demask)
        model_path = transform_model_format(row, source='row', target='path')
        joblib.dump(save_dict, ANA_SAVE_PATH / model_path / f'total_scores{suffix_name}.pkl')

def run_scores_for_each_model(row,suffix_name='', overwrite_config={}, dataset_loading_every_time=False, behav_dt=None, augment_ratio=None,include_data='all',pointwise_loss=False,demask=True):
    model_path = transform_model_format(row, source='row', target='path')
    if os.path.exists(ANA_SAVE_PATH / model_path / f'total_scores{suffix_name}.pkl'):
        pass
        # print(f'total_scores{suffix_name} for {model_path} already exists; skip')
        # continue
    config = transform_model_format(row, source='row', target='config')
    config.update(overwrite_config)
    ag = transform_model_format(config, source='config', target='agent')
    print(model_path)
    if dataset_loading_every_time:
        ## the dataset is loaded every time, but it's slow
        behav_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config)).behav_to(config)
    else:
        # the dataset is loaded only once
        # might introduce bugs if the dataset is changed silently,
        # e.g., the datasets are generated by different agents in the same folder
        if behav_dt is None:
            behav_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config)).behav_to(config)
        if 'include_embedding' in config and behav_dt.include_embedding != config['include_embedding'] or \
                'include_embedding' not in config and hasattr(behav_dt,
                                                              'include_embedding') and behav_dt.include_embedding or \
                'blockinfo' in config and behav_dt.include_block != config['blockinfo'] or \
                'blockinfo' not in config and hasattr(behav_dt, 'include_block') and behav_dt.include_block:
            behav_dt = Dataset(config['dataset'], behav_data_spec=construct_behav_data_spec(config)).behav_to(config)
        if behav_dt.behav_format != config['behav_format']:
            behav_dt = behav_dt.behav_to(config)

    if hasattr(behav_dt, 'augment_ratio'):
        if augment_ratio is None:
            augment_ratio = behav_dt.augment_ratio
        elif augment_ratio != behav_dt.augment_ratio:
            raise ValueError('augment_ratio not consistent')
        else:
            pass
    if include_data == 'all':  # all data in the dataset
        data = behav_dt.get_behav_data(np.arange(behav_dt.batch_size), config)
    elif include_data == 'trainvaltest':
        data = behav_dt.get_behav_data(np.unique(np.concatenate([config['train_index'], config['val_index'], config['test_index']])), config)
    elif include_data == 'test':  # only the test data when the model is trained on other data
        data = behav_dt.get_behav_data(config['test_index'], config)
    elif include_data == 'test_augment':  # test data with augmentation
        data = behav_dt.get_behav_data(behav_dt.get_after_augmented_block_number(config['test_index']), config)
    else:
        raise ValueError('include_data not recognized')

    model_pass = ag.forward(data, standard_output=True, pointwise_loss=pointwise_loss,
                            demask=demask)  # a dict of lists of episodes
    if pointwise_loss:
        model_behav_loss = model_pass['behav_loss']
        total_corrected_trials = model_pass['total_corrected_trials']
    else:
        model_behav_loss = None
        total_corrected_trials = None

    model_scores = model_pass['output']
    model_internal = model_pass['internal']
    if 'mask' in model_pass:
        model_mask = model_pass['mask']
    else:
        model_mask = None

    if isinstance(model_internal, dict):  # for cog model
        if 'state_var' in model_internal:
            model_internal = model_internal['state_var']
        else:
            model_internal = []  # for cog model with no internal state

    if len(model_internal) > 0 and len(model_internal[0]) > 0:
        hid_state = np.concatenate(model_internal, axis=0)
        hid_state_lb = np.min(hid_state, axis=0)
        hid_state_ub = np.max(hid_state, axis=0)
    else:
        hid_state_lb = np.zeros(0)
        hid_state_ub = np.zeros(0)

    os.makedirs(ANA_SAVE_PATH / model_path, exist_ok=True)
    save_dict = {
        'scores': model_scores,
        'internal': model_internal,
        'trial_type': data['trial_type'],
        'hid_state_lb': hid_state_lb,
        'hid_state_ub': hid_state_ub,
        'behav_loss': model_behav_loss,
        'mask': model_mask,
        'augment_ratio': augment_ratio,
        'total_corrected_trials': total_corrected_trials,
        'config': config,
    }
    if 'label' in data:
        save_dict['label'] = data['label']
    return save_dict, behav_dt, augment_ratio



if __name__ == '__main__':
    pass
    # find_best_models_for_exp(exp_folder, 'PRLCog', additional_rnn_keys={'model_identifier_keys': ['input_dim']})
    # model_paths = get_model_path_in_exp(exp_folder)
    # [print(i, m) for i,m in enumerate(model_paths)]
