from setuptools import setup, find_packages

setup(
    name='interface-analyzer',
    version='0.1.0',
    description='Tools for solid-liquid interface detection and Capillary Fluctuation Method (CFM) analysis.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='KaiLiu',
    url='http://yourrepository.com',
    packages=find_packages(), # Automatically finds the interface_analyzer folder
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'matplotlib>=3.4',
        'ovito>=3.7',  # Essential dependency for config processing
        'tqdm',        # Useful for processing loops
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics'
    ],
    python_requires='>=3.8',
)
