# The MIT License (MIT)
# Copyright © 2024 Chakana.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import io
import os
import sys
import copy
import json
import time
import types
import boto3
import torch
import typer
import wandb
import random
import argparse
import tempfile
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from dataset import SubsetFineWebEdu2Loader
from transformers import GPT2Config, GPT2LMHeadModel
env_config = {**dotenv_values(".env"), **os.environ}

def main(
    bucket: str = 'decis',
    aws_access_key_id: str = env_config.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key: str = env_config.get('AWS_SECRET_ACCESS_KEY'),
):
    # Create the hparams item.
    hparams = SimpleNamespace(
        bucket = bucket,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key,
    )
    # Create your S3 connection.
    client: boto3.client = boto3.client(
        's3',
        region_name = 'us-east-1',
        aws_access_key_id = hparams.aws_access_key_id,
        aws_secret_access_key = hparams.aws_secret_access_key
    )
    response = client.list_objects_v2(Bucket = hparams.bucket)
    file_names = [content['Key'] for content in response.get('Contents', [])]
    print(file_names)

# Main function.
if __name__ == "__main__":
    typer.run(main)
