"""
This file should be imported at the beginning of every script if run from console.
It will check the path and change the working directory to the root of the project.
This will make sure that the relative path is correct.
"""
from pathlib import Path
import os

def run():
    path = Path.cwd()
    while len(path.name) and path.name != 'tinyRNN':
        path = path.parent

    if len(path.name):
        os.chdir(path)
    else:
        raise ValueError('Cannot find the root directory of the project.')

run()