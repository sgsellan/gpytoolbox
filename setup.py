# This is essentially lifted from https://raw.githubusercontent.com/nmwsharp/polyscope-py/master/setup.py
# Thanks, Nick!
import os
import re
import sys
import platform
import subprocess
import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

__version__ = '0.0.3'

class CMakeExtension(Extension):
    # Boilerplate that I don't fully understand
    def __init__(self, name, sourcedir='', exclude_arch=False):
        Extension.__init__(self, name, sources=[])
        # Directory where my python package is
        self.sourcedir = os.path.abspath(sourcedir)
        self.exclude_arch = exclude_arch

class CMakeBuild(build_ext):
    def run(self):
        # Check that cmake is installed
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        # If you're Windows, you need a good enough version of Cmake
        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")
        # Call to build (see below)
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]


        # This is horrible, I don't know other way of installing dependencies on the wheel dependencies
        # os.system("python -m pip install numpy")
        # if platform.system() == "Linux":
        #     os.system("apt-get install \
        #       libmpfr-dev \
        #       libgmp-dev")

        # print("=----------------------------=")
        # print(platform.system())
        #cfg = 'Debug' if self.debug else 'Release'
        cfg = 'Release'
        build_args = ['--config', cfg]

        # Setting up call to cmake, platform-dependent
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if not ext.exclude_arch: 
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                else:
                    cmake_args += ['-A', 'Win32']
                build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        if self.distribution.verbose > 0:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']


        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''), self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        if(self.distribution.verbose > 0):
            print("Running cmake configure command: " + " ".join(['cmake', ext.sourcedir] + cmake_args))
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        
        if(self.distribution.verbose > 0):
            print("Running cmake build command: " + " ".join(['cmake', '--build', '.'] + build_args))
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

def main():

    with open('README.md') as f:
        long_description = f.read()

    # Applies to windows only.
    # Normally, we set cmake's -A option to specify 64 bit platform when need (and /m for build), 
    # but these are errors with non-visual-studio generators. CMake does not seem to have an idiomatic 
    # way to disable, so we expose an option here. A more robust solution would auto-detect based on the 
    # generator.  Really, this option might be better titled "exclude visual-studio-settings-on-windows"
    if "--exclude-arch" in sys.argv:
        exclude_arch = True
        sys.argv.remove('--exclude-arch')
    else:
        exclude_arch = False

    setup(
        name='gpytoolbox',
        version=__version__,
        author='Silvia Sellán and Oded Stein',
        author_email='sgsellan@cs.toronto.edu',
        url='https://gpytoolbox.org',
        description='Gpytoolbox: A Python Geometry Processing Toolbox.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        license="MIT",
        package_dir = {'': 'src'},
        packages=setuptools.find_packages(where="src"),
        install_requires=['numpy', 'scipy', 'scikit-image'],
        ext_modules=[CMakeExtension('.', exclude_arch=exclude_arch)],
        setup_requires=['pybind11>=2.4'],
        cmdclass=dict(build_ext=CMakeBuild),
        zip_safe=False,
        test_suite="test",
    )


if __name__ == "__main__":
    main()