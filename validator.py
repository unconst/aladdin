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

import os
import math
import time
import boto3
import torch
import wandb
import argparse
import traceback
import bittensor as bt
from tqdm import tqdm
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Dict, Optional

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
    print('\tBucket:', config.bucket )
    
    # Init model.
    configuration = GPT2Config( output_hidden_states = False, n_positions = config.sequence_length )
    model = GPT2LMHeadModel( config = configuration )
    print ('\tInit Model.')
    
    # Build the tokenizer and optimizer.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( config.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
    tokenizer.pad_token = tokenizer.eos_token        
    print ('\tTokenizer:', config.tokenizer_name)
    
    # Upload my state.
    upload_model(
        wallet = wallet, 
        model = model, 
        extras = { 'sequence_length': config.sequence_length, 'tokenizer_name': config.tokenizer_name, 'loss': 10000000 }, 
        bucket = config.bucket,
        CLIENT = CLIENT
    )
    # (Optional) Clone the base model for comparisions later.
    # We might want to put limits on how far off the base model the updates are.
    # base_model: torch.nn.Module = copy.deepcopy(model.cpu())
    
    # Init weights and biases
    if config.use_wandb:
        wandb.init(project='aladdin', name = f'{wallet.name}-{wallet.hotkey_str}', config = config )
        
    # Remember delta for later removal.
    last_update_block = 0
    best_loss = 15 # Basically Infinity.
    while True:
        try:
            
            # Get metadata from all miner models filtering None values, stale values and thrashing values.
            master_uid = metagraph.S.argmax()
            metadata = { uid: get_metadata( uid, metagraph, subtensor, CLIENT = CLIENT ) for uid in metagraph.uids if master_uid != uid }
            metadata = { uid: meta for uid, meta in metadata.items() if meta is not None } # Filter None values (no model.)
            metadata = { uid: meta for uid, meta in metadata.items() if meta.blocks_since_modified < config.temperature * 10 } # Staleness limit.
            # metadata = { uid: meta for uid, meta in metadata.items() if meta.blocks_since_modified > config.temperature / 10 } # (Optional) Speed limit.

            # Check for no miners active
            if all( meta is None for _, meta in metadata.items() ):
                print("No active miners found. Sleeping for 5 seconds.")
                time.sleep(5)
                continue
            elif config.use_wandb:
                wandb.log({ "n_miners": len(list(metadata.keys())) })
                
            # Load the next dataset pages for eval here making the pages consistent for each miner
            dataset = SubsetFineWebEdu2Loader( 
                batch_size = config.batch_size, 
                sequence_length = config.sequence_length,
                num_pages = config.num_pages, # TODO adapt the number of pages as well.
                tokenizer = tokenizer
            )
            
            # Sort the metadata by older models first.
            start_block = subtensor.block # Record this for consistent reduction for all miners.
            # Run through the models based on last_modified.
            # TODO(const): note that the last modified can change during the eval.
            metadata = dict(sorted(metadata.items(), key=lambda item: item[1].last_modified, reverse=True))
            for next_uid, next_meta in metadata.items():
                
                # Load the model.
                try:
                    model = download_model( metadata = next_meta, device = config.device, CLIENT = CLIENT )
                    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( config.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
                    tokenizer.pad_token = tokenizer.eos_token
                except Exception as e:
                    print(f"Error loading model: {e}")
                    continue
                
                # Iterate over the batches from these pages evaling the model.
                model.eval()
                losses = []
                for _, batch in enumerate(tqdm(dataset, desc="Processing batches", leave=True)):
                    
                    # Shift the input ids to create labels.
                    input_ids = torch.tensor( batch, dtype=torch.long ).to( config.device )
                    labels = input_ids.clone()
                    labels[:, :-1] = input_ids[:, 1:]
                    labels[:, -1] = tokenizer.pad_token_id

                    # Forward pass
                    outputs = model( input_ids=input_ids, labels=labels )
                    losses.append( outputs.loss.item() )
                    
                # Compute the avg loss on all batches from all pages.
                avg_loss = sum( losses ) / len( losses )
                
                # Uses an epsilon: (start_block - last_update_block)/temperature reduction.
                # because we run the models in order of upload this puts the oldest model at an advantage.
                # The epsilon decays slowly until it hits 0 as the temperature term.
                # As time progresses the temperature gets pushed further and further out simulating a slower annealing.
                start_epsilon = 0.1 # Decreases from 0.1 -> 0 over the temperature period.
                block_epsilon = max(0, start_epsilon - start_epsilon * (start_block - last_update_block) / config.temperature )
                threshold = best_loss - best_loss * block_epsilon
                print (f'UID:{next_meta.uid}, Filename:{next_meta.filename}, BestLoss:{best_loss}, AvgLoss: {avg_loss}, Epsilon: {block_epsilon}, Threshold: {threshold}, ')
                if config.use_wandb:
                    wandb.log({ "AvgLoss": avg_loss, 'Epsilon': block_epsilon, 'Threshold': threshold, 'BestLoss': best_loss })
                
                # If the average loss is less than the threshold give all incentive to this miner and upload the new state.                          
                if avg_loss < threshold:
                    best_loss = avg_loss
                    
                    # Make the epsilon decay period equivalent to the duration it took to improve the loss x 2.
                    # We use the start_block here rather than subtensor.block since this is when we started evaling the models.
                    # This will increase the temperature if we took a long time to beat the epsilon.
                    config.temperature = (start_block - last_update_block) * 2 # Doubling works well.
                    last_update_block = start_block 
                    
                    # Upload the new best model this is then pulled by all the miners.
                    upload_model(
                        wallet = wallet, 
                        model = model, 
                        extras = { 'sequence_length': next_meta.sequence_length, 'tokenizer_name': next_meta.tokenizer_name, 'loss': avg_loss }, 
                        bucket = config.bucket,
                        CLIENT = CLIENT
                    )
                    # Set weights to the miner who beat the epsilon first.
                    subtensor.set_weights(
                        wallet = wallet,
                        netuid = config.netuid,
                        uids = [ next_uid ],
                        weights = [ 1.0 ], # TODO (const): there might be something better than this possibly using a moving average?
                        wait_for_inclusion = False,
                        wait_for_finalization = False
                    )  
                    if config.use_wandb:
                        wandb.log({ "BestUID": next_uid, "Temperature": config.temperature })
                        
                    # Break the loop here. This gives the earlier miners the advantage.
                    break
                

                                            
        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            to_delete = get_metadata( my_uid, metagraph, subtensor, CLIENT = CLIENT )
            if to_delete != None:
                CLIENT.delete_object( Bucket = config.bucket, Key = to_delete.filename )
            break
        
        # Handle unknown exceptions, continue training after 5 seconds.
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            time.sleep(5)
            continue

# Main function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Miner script')
    parser.add_argument('--name', type=str, default='miner', help='Name of the miner')
    parser.add_argument('--netuid', type=int, default=1, help='Bittensor network uid.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--temperature', type=int, default=100, help='Epsilon half life.')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=2048, help='Sequence Length.')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Tokenizer name.')
    parser.add_argument('--num_pages', type=int, default=5, help='Number of pages to load')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    main( config ) 