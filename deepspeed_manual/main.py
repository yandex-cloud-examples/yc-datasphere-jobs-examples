from xllm import Config
from xllm.datasets import GeneralDataset
from xllm.cli import cli_run_train

import deepspeed

print(deepspeed.__file__, deepspeed.__version__)

if __name__ == '__main__':
    train_data = ["Hello!"] * 100
    train_dataset = GeneralDataset.from_list(data=train_data)
    cli_run_train(config_cls=Config, train_dataset=train_dataset)
