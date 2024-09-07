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
import boto3
import torch
import wandb
import typer
import argparse
import tempfile
import bittensor as bt
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, List, Optional, Tuple

# Import common tooling.
from common import upload_model, get_metadata, download_model, SubsetFineWebEdu2Loader

# Instantiate my S3 client.
env_config = {**dotenv_values(".env"), **os.environ}
AWS_ACCESS_KEY_ID = env_config.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = env_config.get('AWS_SECRET_ACCESS_KEY')
CLIENT: boto3.client = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_ACCESS_KEY
)

# Main function.
def main( config ):
    print ( config )
    
    # Init Bittensor objects.
    wallet = bt.wallet( config = config )
    subtensor = bt.subtensor( config = config )
    metagraph = subtensor.metagraph( netuid = config.netuid )
    print ('\tWallet', wallet)
    print ('\tSubtensor', subtensor)
    print ('\tMetagraph', metagraph)

    # Get my registered UID.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        raise ValueError(f'Wallet {wallet} is not registered on subnet: {metagraph.netuid}')
    my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
    print ( '\tUID:', my_uid )
    
    # Assert the chain commitment.
    try:
        if config.bucket != subtensor.get_commitment(config.netuid, my_uid):
            raise ValueError(f'Chain commitment does not match: {config.bucket}')
    except Exception:
        subtensor.commit( wallet, config.netuid, config.bucket)
    print('\tBucket:', config.bucket , '\n')
                    
    # Init weights and biases
    if config.use_wandb:
        if config.resume:
            wandb.init(project='aladdin', resume='allow', name = f'{config.name}-{wallet.name}-{wallet.hotkey_str}', config = config )
        else:
            wandb.init(project='aladdin', name = f'{config.name}-{wallet.name}-{wallet.hotkey_str}', config = config )
        
    # Main training loop.
    model = None
    current = None
    while True:
        try:
            
            # Get metadata from the master (the key with the most amount of stake.)
            # Descends the stake list until it finds a master with stake.
            master = None
            for uid in reversed(np.argsort(metagraph.S)):
                master = get_metadata(uid, metagraph, subtensor, CLIENT = CLIENT)
                if master is not None:
                    break
            if master is None:
                print("Waiting for the master to upload the model. Sleeping for 5 seconds.")
                time.sleep(5)    
                continue
            if config.use_wandb:
                wandb.log({ "Master": master.uid } )
            
            # If the master has a newer model, download it.
            if current == None or master.last_modified > current.last_modified:
                model = download_model( metadata = master, device = config.device, CLIENT = CLIENT )
                tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( master.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
                tokenizer.pad_token = tokenizer.eos_token    
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr = config.learning_rate,  # Peak learning rate
                    betas = ( config.optimizer_beta1, config.optimizer_beta2 ), # B1 and B2
                    weight_decay = config.optimizer_weight_decay  # Weight decay
                )
                current = master
                
            # Load training dataset pages.
            dataset = SubsetFineWebEdu2Loader( 
                batch_size = config.batch_size, 
                sequence_length = current.sequence_length,
                num_pages = config.num_pages, 
                tokenizer = tokenizer
            )
                            
            # Iterate over the batches from these pages training the model.
            model.train()
            for _, batch in enumerate(tqdm(dataset, desc="Processing batches", leave=True)):
                
                # Shift the input ids to create labels.
                input_ids = torch.tensor(batch, dtype=torch.long).to(config.device)
                labels = input_ids.clone()
                labels[:, :-1] = input_ids[:, 1:]
                labels[:, -1] = tokenizer.pad_token_id

                # Forward pass
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss

                # Accumulate the gradients.
                loss.backward()
                
                # Step the optimizer
                optimizer.step()
                optimizer.zero_grad()
                
                # Log to wandb.
                tqdm.write(f"Loss: {loss.item()}")
                if config.use_wandb:
                    wandb.log({ "Loss": loss.item(), "Perplexity": math.exp(loss.item()) } )
                        
            # Upload the latest update.
            upload_model( 
                model = model, 
                wallet = wallet, 
                bucket = config.bucket,
                extras = {},
                CLIENT = CLIENT
            )
                        
        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            to_delete = get_metadata( my_uid, metagraph, subtensor, CLIENT = CLIENT )
            if to_delete != None:
                CLIENT.delete_object( Bucket = config.bucket, Key = to_delete.filename )
            if config.use_wandb:
                wandb.finish()
                api = wandb.Api()
                run = api.run(f"{wandb.run.entity}/{wandb.run.project}/{wandb.run.id}")
                run.delete()
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print (f"Error: {e}")
            time.sleep(5)
            continue

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default='miner', help='Optional miner name')
    parser.add_argument('--netuid', type=int, default=1, help='Bittensor network uid.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer')
    parser.add_argument('--optimizer_beta1', type=float, default=0.9, help='Beta1 for the optimizer')
    parser.add_argument('--optimizer_beta2', type=float, default=0.95, help='Beta2 for the optimizer')
    parser.add_argument('--optimizer_weight_decay', type=float, default=0.1, help='Weight decay for the optimizer')
    parser.add_argument('--num_pages', type=int, default=5, help='Number of pages to load')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    main( config ) 