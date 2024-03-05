from setuptools import setup, find_packages

setup(
    name='ehrsql-2024',
    version='0.1',
    packages=find_packages(include=['preprocess', 'scoring_program', 'utils', 'ingestion_program']),
    install_requires=[
        'sqlparse==0.4.4',
        'scikit-learn==1.4.1.post1', 
        'transformers==4.38.2',
        'peft==0.9.0',
        'trl==0.7.11',
        'wandb==0.16.3',
        'xformers==0.0.24',
        'accelerate==0.27.2',
        'bitsandbytes==0.42.0',
        # 'unsloth[conda] @ git+https://github.com/unslothai/unsloth.git@February-Gemma-2024'
    ]
)
