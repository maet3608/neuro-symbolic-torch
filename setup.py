import os
import sys
import glob
import shutil
import nstorch

from setuptools import setup, find_packages, Command
from setuptools.command.test import test as TestCommand


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for folder in ['build', 'dist']:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        for egg_file in glob.glob('*egg-info'):
            shutil.rmtree(egg_file)


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    name='torchy',
    version=nstorch.__version__,
    url='https://github.com/maet3608/neuro-symbolic-torch',
    download_url='https://github.com/maet3608/neuro-symbolic-torch',
    license='Apache Software License (http://www.apache.org/licenses/LICENSE-2.0)',
    author='Stefan Maetschke',
    author_email='stefan.maetschke@gmail.com',
    description='Neuro-symbolic deep learning framework based on Pytorch',
    install_requires=[
        'nutsml >= 1.0.42',
        'torch >= 1.0.0',
        'pytest >= 3.0.3',
    ],
    tests_require=['pytest >= 3.0.3'],
    platforms='any',
    packages=find_packages(exclude=['setup']),
    include_package_data=True,
    cmdclass={
        'test': PyTest,
        'clean': CleanCommand,
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
