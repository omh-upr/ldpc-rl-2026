#!/usr/bin/env python3
"""
Checks that a dataset is judged by the arrays training consumes.

`validate_existing_dataset` decides whether a file already on disk is the
dataset a config asked for. If it says yes wrongly, the run trains on data
nobody requested and says nothing about it.

It used to read `splits.attrs['num_test_samples']` and never open
`train_indices` or `test_indices`, which training and evaluation actually use,
so a split could be missing, overlapping, truncated or drawn from the wrong
phases and still be accepted. It also checked the codeword width of `phase_0`
only, and only when the caller passed a code.

Each check below damages a copy of a real generated dataset in one way and
asserts the damage is refused. The undamaged copy must still be accepted, which
is the half that matters most: a validator that rejects everything is no use.

Run:
    PYTHONPATH=. python tests/verify_dataset_validation.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

from src.core.ldpc_code import LDPC_Code
from src.data.create_dataset import create_ldpc_dataset, validate_existing_dataset

CODE_ID = '24x48'
TEST_SIZE = 0.3
PHASES = [(0, 12), (1, 8)]   # (num_errors, num_samples) per curriculum phase


def write_config(path):
    """A two-phase curriculum small enough to generate in seconds."""
    config = {
        'seed': 42,
        'dataset_name': 'validation_probe',
        'uniqueness': {'scope': 'dataset',
                       'ensure_unique_messages': False,
                       'ensure_unique_error_patterns': False},
        'curriculum': [
            {'phase_name': f'Phase{index + 1}', 'num_samples': samples,
             'error_models': [{'type': 'uniform_count', 'weight': 1.0,
                               'params': {'num_errors': errors}}]}
            for index, (errors, samples) in enumerate(PHASES)
        ],
    }
    path.write_text(json.dumps(config, indent=2))
    return config


def judge(dataset, config, code=None, test_size=TEST_SIZE):
    return validate_existing_dataset(dataset, config, test_size=test_size, code=code)


def damaged(original, config, name, damage, code=None):
    """Apply one damage to a copy and return the verdict on it."""
    copy = original.parent / f'{name}.h5'
    shutil.copy(original, copy)
    with h5py.File(copy, 'a') as f:
        damage(f)
    return judge(copy, config, code=code)


def check_accepts_the_real_thing(dataset, config, code):
    print("1. the dataset that was actually generated is accepted")
    ok, reason = judge(dataset, config, code=code)
    assert ok, f"rejected a valid dataset: {reason}"
    ok, reason = judge(dataset, config, code=code, test_size=None)
    assert ok, f"rejected a valid dataset when no split was requested: {reason}"
    print(f"   accepted with and without a requested split ({reason})")


def check_split_arrays(dataset, config, code):
    print("2. the split is judged by its index arrays, not its count attribute")

    total = sum(samples for _, samples in PHASES)
    expected_test = sum(s - int(s * (1 - TEST_SIZE)) for _, s in PHASES)

    def delete_test_indices(f):
        del f['splits/test_indices']          # count attribute left in place

    def test_from_training(f):
        train = f['splits/train_indices'][:]
        del f['splits/test_indices']
        f['splits'].create_dataset('test_indices', data=train[:expected_test])

    def truncate_test(f):
        test = f['splits/test_indices'][:]
        del f['splits/test_indices']
        f['splits'].create_dataset('test_indices', data=test[:1])

    def out_of_bounds(f):
        test = f['splits/test_indices'][:]
        test[0] = total + 5
        f['splits/test_indices'][:] = test

    def duplicate_in_test(f):
        test = f['splits/test_indices'][:]
        test[1] = test[0]
        f['splits/test_indices'][:] = test

    def all_test_from_one_phase(f):
        # Right number of test samples, all drawn from phase 0: the total check
        # passes and stratification is silently gone.
        train = f['splits/train_indices'][:]
        test = f['splits/test_indices'][:]
        every = np.sort(np.concatenate([train, test]))
        new_test = every[:expected_test]
        new_train = every[expected_test:]
        del f['splits/test_indices'], f['splits/train_indices']
        f['splits'].create_dataset('test_indices', data=new_test)
        f['splits'].create_dataset('train_indices', data=new_train)

    cases = [
        ('deleted_test', delete_test_indices, 'no test_indices'),
        ('test_from_train', test_from_training, 'test indices copied from train'),
        ('truncated_test', truncate_test, 'one test sample instead of many'),
        ('out_of_bounds', out_of_bounds, 'an index past the end of the dataset'),
        ('duplicate', duplicate_in_test, 'the same sample twice'),
        ('unstratified', all_test_from_one_phase, 'every test sample from one phase'),
    ]
    for name, damage, description in cases:
        ok, reason = damaged(dataset, config, name, damage, code=code)
        assert not ok, f"accepted a dataset with {description}"
        print(f"   refused {description}: {reason}")


def check_widths(dataset, config, code):
    print("3. every phase's width is checked, not only the first")

    def narrow_second_phase(f):
        rows = f['phase_1/noisy_codewords'].shape[0]
        data = f['phase_1/noisy_codewords'][:, :-1]
        del f['phase_1/noisy_codewords']
        f['phase_1'].create_dataset('noisy_codewords', data=data)
        assert f['phase_1/noisy_codewords'].shape == (rows, code.n - 1)

    # Caught even with no code to compare against, because the phases disagree.
    ok, reason = damaged(dataset, config, 'narrow_no_code', narrow_second_phase)
    assert not ok, "accepted phases of different widths"
    print(f"   refused without a code: {reason}")

    ok, reason = damaged(dataset, config, 'narrow_with_code', narrow_second_phase, code=code)
    assert not ok, "accepted phases of different widths"
    print(f"   refused with a code: {reason}")

    # A whole dataset at the wrong width is only detectable against a code.
    other = LDPC_Code.from_default('56x80')
    ok, reason = judge(dataset, config, code=other)
    assert not ok, "accepted a dataset built for another code"
    print(f"   refused a dataset built for another code: {reason}")


def main():
    print("Verifying dataset reuse validation\n")
    workspace = Path(tempfile.mkdtemp(prefix='verify_dataset_'))
    try:
        config = workspace / 'curriculum.json'
        write_config(config)
        dataset = workspace / 'probe.h5'
        code = LDPC_Code.from_default(CODE_ID)

        create_ldpc_dataset(str(config), str(dataset), test_size=TEST_SIZE,
                            code_id=CODE_ID)
        print()

        check_accepts_the_real_thing(dataset, config, code)
        check_split_arrays(dataset, config, code)
        check_widths(dataset, config, code)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print("\nAll checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
