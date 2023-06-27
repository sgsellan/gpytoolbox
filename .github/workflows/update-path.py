import os
import pathlib

# Get the absolute path of the build directory
build_dir = pathlib.Path('.').absolute() / 'build'

# Iterate over all subdirectories in the build directory
for sub_dir in build_dir.rglob('*'):
    if sub_dir.is_dir():
        # Check if the directory contains a .dll file
        if any(file.suffix == '.dll' for file in sub_dir.iterdir()):
            # Add the directory to the PATH
            os.environ['PATH'] = f"{sub_dir.absolute()};{os.environ['PATH']}"
