from setuptools import setup, find_packages

setup(
    name='ehrsql-2024-willystumblr',
    version='0.1',
    packages=find_packages(include=['preprocess', 'scoring_program', 'utils', 'ingestion_program']),
    install_requires=[
        'pyarrow==14.0.2', 
        'transformers==4.37.2',
        'peft==0.8.2',
        'trl==0.7.10',
        'wandb==0.16.3',
        'torchaudio==2.1.0',
        'torchvision==0.16.0',
        'xformers==0.0.24',
        'bitsandbytes==0.42.0',
        'unsloth[conda] @ git+https://github.com/unslothai/unsloth.git'
    ]
)