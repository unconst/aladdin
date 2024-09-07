
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
import torch
import hashlib
import tempfile
import bittensor as bt
from types import SimpleNamespace
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
from typing import Dict, List, Optional, Tuple

from .dataset import SubsetFineWebEdu2Loader

def hash_model( module: torch.nn.Module ) -> str:
    """
    Generates a SHA-256 hash of the model's state dictionary.

    This function iterates through the model's state dictionary, concatenates the byte representation
    of each parameter, and then generates a SHA-256 hash of this concatenated byte string.

    Args:
        model (torch.nn.Module): The model to hash.

    Returns:
        str: The SHA-256 hash of the model's state dictionary.
    """
    # Extract the state dictionary from the module which contains all the parameters.
    module_state_dict = module.state_dict()
    
    # Concatenate all the model state values into a single byte string.
    concatenated_model_states_bytes = b''.join(
        [value.cpu().numpy().tobytes() for value in module_state_dict.values()]
    )
    
    # Generate a SHA-256 hash from the concatenated bytes.
    module_hash = hashlib.sha256(concatenated_model_states_bytes).hexdigest()
    return module_hash

def get_metadata( 
        uid, 
        metagraph, 
        subtensor, 
        CLIENT 
    ) -> Optional[ SimpleNamespace ]:
    """
    Retrieves metadata for a specified model from a storage service.

    Args:
        uid (int): The unique identifier for the model.
        metagraph: The bittensor metagraph containing network information.
        subtensor: The bittensor subtensor object used to interact with the network.
        CLIENT: The client used to interact with the storage service.

    Returns:
        Optional[SimpleNamespace]: A namespace containing the metadata if successful, otherwise None.
    """
    try:
        # Get the bucket name using the subtensor and metagraph information
        bucket = subtensor.get_commitment(metagraph.netuid, uid)
        
        # Define the filenames for the model and its metadata
        filename = f"model-{metagraph.hotkeys[uid]}.pt"
        metadata_filename = f"model-{metagraph.hotkeys[uid]}_metadata.json"
        
        # Get the metadata of the model file from the storage service
        response = CLIENT.head_object(Bucket=bucket, Key=filename)
        
        # Extract and calculate metadata information
        metadata = {}
        metadata['last_modified'] = int(response['LastModified'].timestamp())
        metadata['blocks_since_modified'] = int((time.time() - int(response['LastModified'].timestamp())) / 12)
        metadata['bucket'] = bucket
        metadata['filename'] = filename
        metadata['uid'] = int(uid)
        
        # Get the metadata file from the storage service
        metadata_response = CLIENT.get_object(Bucket=bucket, Key=metadata_filename)
        
        # Read and update the metadata with the content of the metadata file
        metadata_json = json.loads(metadata_response['Body'].read().decode('utf-8'))
        metadata.update(metadata_json)
        
        # Return the metadata as a SimpleNamespace object
        return SimpleNamespace(**metadata)
    except Exception as e:
        # Return None if any exception occurs
        return None
    
def download_model( 
        metadata: SimpleNamespace, 
        device: str, 
        CLIENT 
    ) -> Optional[ torch.nn.Module ]:
    """
    Downloads a model from a specified bucket and loads it onto the specified device.

    Args:
        metadata (SimpleNamespace): Metadata containing information about the model to be downloaded.
        device (str): The device to load the model onto (e.g., 'cpu' or 'cuda').
        CLIENT: The client used to interact with the storage service.

    Returns:
        Optional[torch.nn.Module]: The downloaded model if successful, otherwise None.
    """
    print(f'Downloading model from {metadata.filename}@{metadata.bucket}')  # Log the start of the download process
    start_time = time.time()  # Record the start time for the download

    # Check the model type and initialize the appropriate model configuration and model
    if metadata.model_type == "llama":
        model_config = LlamaConfig(**metadata.model_config)  # Create Llama model configuration
        model = LlamaForCausalLM(model_config)  # Initialize Llama model
    if metadata.model_type == "gpt2":
        model_config = GPT2Config(**metadata.model_config)  # Create GPT-2 model configuration
        model = GPT2LMHeadModel(model_config)  # Initialize GPT-2 model

    # Download the model file from the storage service
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        CLIENT.download_fileobj(metadata.bucket, metadata.filename, temp_file)  # Download the model file to a temporary file
        temp_file_path: str = temp_file.name  # Get the path of the temporary file
        new_model_state_dict = torch.load(temp_file_path, map_location=torch.device(device), weights_only=True)  # Load the model state dict from the temporary file
        model.load_state_dict(new_model_state_dict)  # Load the state dict into the model

    model.to(device)  # Move the model to the specified device
    os.unlink(temp_file_path)  # Delete the temporary file

    # Log the completion of the download process with the time taken
    print(f"Downloaded model from {metadata.filename}@{metadata.bucket} in {time.time() - start_time} seconds.")
    return model  # Return the downloaded model

def upload_model( 
        wallet: 'bt.wallet',
        model: torch.nn.Module, 
        extras: Dict[ str, object ],
        bucket: str,
        CLIENT,
    ):
    """
    Uploads a model to a specified bucket along with its metadata.

    Args:
        wallet (bt.wallet): The wallet containing the hotkey used to generate the filename.
        model (torch.nn.Module): The model to be uploaded.
        extras (Dict[str, object]): Additional metadata to be uploaded with the model.
        bucket (str): The bucket to upload the model to.
        CLIENT: The client used to interact with the storage service.

    Returns:
        None
    """
    start_time = time.time()  # Record the start time for the upload process
    model_state_dict = model.state_dict()  # Get the state dictionary of the model

    # Extract the configuration from the model and update extras with model type and configuration
    if isinstance(model, LlamaForCausalLM):
        config = model.config  # Get the configuration of the Llama model
        extras.update({
            'model_type': 'llama',  # Add model type to extras
            'model_config': config.to_dict()  # Add model configuration to extras
        })
    elif isinstance(model, GPT2LMHeadModel):
        config = model.config  # Get the configuration of the GPT-2 model
        extras.update({
            'model_type': 'gpt2',  # Add model type to extras
            'model_config': config.to_dict()  # Add model configuration to extras
        })
    # Add model hashes.
    extras['model_has'] = hash_model( model )

    # Generate filenames for the model and its metadata
    filename = f'model-{wallet.hotkey.ss58_address}.pt'  # Filename for the model
    metadata_filename = f"model-{wallet.hotkey.ss58_address}_metadata.json"  # Filename for the metadata

    # Upload the metadata to the storage service
    metadata_buffer = io.BytesIO(json.dumps(extras).encode('utf-8'))  # Create a buffer for the metadata
    CLIENT.upload_fileobj(metadata_buffer, bucket, metadata_filename)  # Upload the metadata buffer to the storage service

    # Grant read and list permissions to all users for the metadata
    CLIENT.put_object_acl(
        Bucket=bucket,
        Key=metadata_filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantList='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
    )

    # Upload the model to the storage service
    with io.BytesIO() as module_buffer:
        torch.save(model_state_dict, module_buffer)  # Save the model state dictionary to the buffer
        module_buffer.seek(0)  # Reset the buffer's position to the beginning
        CLIENT.upload_fileobj(module_buffer, bucket, filename)  # Upload the model buffer to the storage service

    # Grant read and list permissions to all users for the model
    CLIENT.put_object_acl(
        Bucket=bucket,
        Key=filename,
        GrantRead='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantReadACP='uri="http://acs.amazonaws.com/groups/global/AllUsers"',
        GrantList='uri="http://acs.amazonaws.com/groups/global/AllUsers"'
    )

    # Log the completion of the upload process with the time taken
    print(f"Uploaded model to {filename}@{bucket} in {time.time() - start_time} seconds.")