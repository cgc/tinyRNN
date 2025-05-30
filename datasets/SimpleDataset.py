import pathlib

import torch
import joblib
import numpy as np

from .BaseTwoStepDataset import BaseTwoStepDataset

class SimpleDataset:
    def __init__(self, data_path=None, behav_data_spec=None, neuro_data_spec=None, verbose=True):
        '''
        We expect a behav_data_spec that specifies a dataset and format, as follows:
        {
            'dataset': 'Simple',
            'behav_format': 'tensor',
            'behav_data_spec': {
                'data': {
                    'action': [np.array([...]), np.array([...])],
                    'reward': [...],
                    ...,
                },
                'input_format': [
                    dict(name='action', one_hot_classes=3),
                    dict(name='reward'),
                ],
                'output_dim': 3, # number of output dimensions
                'target_name': 'action',
            },
            ...
        }
        '''
        assert data_path == '' or data_path == pathlib.Path(''), data_path
        self.data = joblib.load(behav_data_spec['data'])
        self.input_format = behav_data_spec['input_format']

        self.output_dim = behav_data_spec['output_dim']
        self.target_name = behav_data_spec['target_name']
        self.input_dim = sum([
            fmt.get('one_hot_classes', 1) # By default, values take one dimension.
            for fmt in self.input_format
        ])

        arbitrary_key = list(self.data.keys())[0]
        self.batch_size = len(self.data[arbitrary_key])
        self.trial_counts = [len(t) for t in self.data[arbitrary_key]]

        for fmt in self.input_format:
            key = fmt['name']
            values = self.data[key]

            # Validation of sizes
            assert len(values) == self.batch_size, (key, len(values), self.batch_size)
            assert [len(t) for t in values] == self.trial_counts, (key, [len(t) for t in values], self.trial_counts)

            # Validation of formatting
            if 'one_hot_classes' in fmt:
                for t in values:
                    assert np.all((0 <= t) & (t < fmt['one_hot_classes'])), f'Expected values in [0, {fmt["one_hot_classes"]}) for key {key} but found unique values {np.unique(t)}'

        print(f'Total batch size: {self.batch_size}')

        # Inputs for compatibility
        self.torch_beahv_input = None
        self.behav = {}

    def load_data(self, behav_data_spec, neuro_data_spec=None, verbose=True):
        '''
        While this seems like part of the public API, it is not used in training.py
        '''
        raise NotImplementedError

    def behav_to(self, format_config=None):
        assert format_config != 'cog_session'
        return BaseTwoStepDataset.behav_to(self, format_config)

    def _behav_to_tensor(self, format_config):
        """Transform standard behavioral format to tensor format, stored in torch_beahv_* attribute.

        standard format (list of 1d array) -> tensor format (2d array with 0 padding).
        The attributes are:
            torch_beahv_input: tensor format of agent's input
            torch_beahv_input_1hot: tensor format of agent's input (one-hot encoding of trial observations)
            torch_beahv_target: tensor format of agent's target output
            torch_beahv_mask: tensor format of agent's mask (1 for valid trials, 0 for padding trials)

        Not use nan padding:
            rnn model make all-nan output randomly (unexpected behavior, not sure why)
            the one_hot function cannot accept nan
            long type is required for cross entropy loss, but does not support nan value

        Args:
            format_config: A dict specifies how the standard data should be transformed.

        """
        if self.torch_beahv_input is not None:
            return

        trial_num = max(self.trial_counts)

        inputs = []
        for fmt in self.input_format:
            size = fmt.get('one_hot_classes', 1)
            inputs.append(np.zeros((self.batch_size, trial_num, size)))

        target = np.zeros((self.batch_size, trial_num))
        mask = np.zeros((self.batch_size, trial_num))
        # TODO handle target mask

        for b in range(self.batch_size):
            trial_count = self.trial_counts[b]
            idxs = np.arange(trial_count)
            # Default mask to 1
            mask[b, :trial_count] = 1
            # Write data
            for fmt, arr in zip(self.input_format, inputs):
                value = self.data[fmt['name']][b]
                assert len(value) == trial_count
                if 'one_hot_classes' in fmt:
                    # Then we must one-hot code data
                    arr[b, idxs, value] = 1
                else:
                    # Otherwise we just write data in
                    arr[b, :trial_count, 0] = value
                # Invalid value of -1 removes a trial
                invalid = value == -1
                mask[b, invalid] = 0
            # Write target
            target[b, :trial_count] = self.data[self.target_name][b]

        # target_mask *= 9999

        include_embedding = 'include_embedding' in format_config and format_config['include_embedding']
        self.include_embedding = include_embedding
        assert not include_embedding, 'for now, waiting on embedding support'

        assert format_config['output_h0'], 'We assume output_h0 since it simplifies tracking here.'

        device = 'cpu' if 'device' not in format_config else format_config['device']
        input = torch.cat([
            torch.from_numpy(np.swapaxes(i, 0,1)).to(device=device)
            for i in inputs
        ], -1)

        target = torch.from_numpy(np.swapaxes(target, 0,1)).to(device=device)
        mask = torch.from_numpy(np.swapaxes(mask, 0,1)).to(device=device)
        # target_mask = torch.from_numpy(np.swapaxes(target_mask, 0,1)).to(device=device)  # target_mask shape: 2*trial_num, batch_size, act_num

        self.torch_beahv_input = input.double()
        self.torch_beahv_target = target.long()
        # self.torch_beahv_mask = (mask.double(), target_mask.double())
        self.torch_beahv_mask = mask.double()

    def get_behav_data(self, select_indices, format_config=None, selected_trial_indices=None):
        return BaseTwoStepDataset.get_behav_data(self, select_indices, format_config=format_config, remove_padding_trials=True, selected_trial_indices=selected_trial_indices)

    @property
    def total_trial_num(self):
        return np.sum(self.trial_counts)
