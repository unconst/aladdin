
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
import math
import time
import json
import boto3
import torch
import wandb
import typer
import pickle
import base64
import random
import argparse
import tempfile
import bittensor as bt
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from dataset import SubsetFineWebEdu2Loader
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, List, Optional, Tuple

# Encode extras using pickle and base64 to ensure they are strings
def encode_extras(extras: Dict[str, object]) -> Dict[str, str]:
    return {key: base64.b64encode(pickle.dumps(value)).decode('utf-8') for key, value in extras.items()}

# Decode extras using base64 and pickle
def decode_extras(metadata: Dict[str, str]) -> Dict[str, object]:
    return {key: pickle.loads(base64.b64decode(value.encode('utf-8'))) for key, value in metadata.items()}

def get_metadata( uid, metagraph, subtensor, CLIENT ) -> Optional[ SimpleNamespace ]:
    try:
        bucket = subtensor.get_commitment( metagraph.netuid, uid )
        filename = f"model-{metagraph.hotkeys[uid]}.pt"
        response = CLIENT.head_object( Bucket = bucket, Key = filename )
        metadata = {key: value for key, value in response['Metadata'].items()}
        metadata = decode_extras(metadata)
        metadata['last_modified'] = int(response['LastModified'].timestamp())
        metadata['blocks_since_modified'] = int( (time.time() - int(response['LastModified'].timestamp())) / 12 )
        metadata['bucket'] = bucket
        metadata['filename'] = filename
        metadata['uid'] = int(uid)
        return SimpleNamespace( **metadata )
    except Exception as e:
        return None
    
def download_model( metadata: SimpleNamespace, device: str, CLIENT ) -> Optional[ torch.nn.Module ]:
    print (f'Downloading model from {metadata.filename}@{metadata.bucket}')
    start_time = time.time()
    model = GPT2LMHeadModel( GPT2Config( n_positions = int(metadata.sequence_length) )) 
    with tempfile.NamedTemporaryFile( delete = False ) as temp_file:
        CLIENT.download_fileobj( metadata.bucket, metadata.filename, temp_file )            
        temp_file_path: str = temp_file.name
        new_model_state_dict = torch.load( temp_file_path, map_location=torch.device(device), weights_only=True )
        model.load_state_dict(new_model_state_dict)
    model.to(device)
    os.unlink(temp_file_path)
    print(f"Downloaded model from {metadata.filename}@{metadata.bucket} in {time.time() - start_time} seconds.")
    return model

def upload_model( 
        wallet: 'bt.wallet',
        model: torch.nn.Module, 
        extras: Dict[ str, object ],
        bucket: str,
        CLIENT,
    ):
    start_time = time.time()
    model_state_dict = model.state_dict()
    filename = f'model-{wallet.hotkey.ss58_address}.pt'
    print (f'Uploading model to {filename}@{bucket}')
    with io.BytesIO() as module_buffer:
        torch.save(model_state_dict, module_buffer)
        module_buffer.seek(0)
        CLIENT.upload_fileobj(
            module_buffer, 
            bucket, 
            filename, 
            ExtraArgs = {"Metadata": encode_extras( extras )}
        )
    print(f"Uploaded model to {filename}@{bucket} in {time.time() - start_time} seconds.")