import sys
import os
# Path to where the bindings live
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
if os.name == 'nt': # if Windows
    # handle default location where VS puts binary
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Release")))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Debug")))
    try:  # to use Python 3.8's DLL handling
        os.add_dll_directory(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Release")))
        # os.add_dll_directory(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Debug")))
    except AttributeError:  # <3.8, use PATH
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Release"))
        # os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "Debug"))
else:
    # normal / unix case
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build")))
import gpytoolbox
import numpy
import unittest