from setuptools import setup, find_packages

setup(name='gym_platform',
    version='0.1.0',
    install_requires=['gym'],  # And any other dependencies foo needs
    packages=[package for package in find_packages()
                if package.startswith('gym')],
    )