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
import numpy as np
import bittensor as bt
from tqdm import tqdm
from collections import deque
from dotenv import dotenv_values
from types import SimpleNamespace
from transformers import AutoTokenizer
from transformers import GPT2Config, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaConfig, LlamaTokenizer
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
    
    # Build the tokenizer.
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained( config.tokenizer_name, verbose=False, clean_up_tokenization_spaces=True )
    tokenizer.pad_token = tokenizer.eos_token        
    print ('\tTokenizer:', config.tokenizer_name)

    # Init model based on type.
    if not config.resume:
        # If we are not resuming (i.e. new run) We create and upload the initial model.
        if config.model_type == 'gpt2':
            model = GPT2LMHeadModel( config = GPT2Config(
                output_hidden_states = False, 
                n_positions = config.sequence_length
            ))
        elif config.model_type == 'llama':
            model = LlamaForCausalLM( config = LlamaConfig(
                vocab_size = tokenizer.vocab_size,     
                hidden_size = 2040,   
                num_hidden_layers = 12,  
                num_attention_heads = 12,
                intermediate_size = 6144
            ))
        print (f'\tModel: {config.model_type}')
        upload_model(
            wallet = wallet, 
            model = model, 
            extras = { 'sequence_length': config.sequence_length, 'tokenizer_name': config.tokenizer_name }, 
            bucket = config.bucket,
            CLIENT = CLIENT
        )
    
    # Init weights and biases
    if config.use_wandb:
        if config.resume:
            wandb.init(project='aladdin', resume='allow', name = f'{config.name}-{wallet.name}-{wallet.hotkey_str}', config = config )
        else:
            wandb.init(project='aladdin', name = f'{config.name}-{wallet.name}-{wallet.hotkey_str}', config = config )
        
    # Remember delta for later removal.
    last_update_block = 0
    best_loss = 15 # Basically Infinity.
    weights = torch.zeros( (metagraph.n), dtype=torch.float32)
    while True:
        try:
            
            # Resync bittensor objects to get latest state.
            subtensor = bt.subtensor( config = config )
            metagraph = subtensor.metagraph( netuid = config.netuid )
            if metagraph.n > len(weights):
                weights = torch.cat((weights, torch.zeros((metagraph.n - len(weights)), dtype=torch.float32)))
            
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
            metadata = deque(sorted(metadata.items(), key=lambda item: item[1].last_modified, reverse=True))
            while metadata:
                
                # Pull the next metadata off the queue.
                # If the last modified has changed. Shuttle it to the back of the queue.
                # This forces us to run the models in the order they were uploaded.
                next_uid, next_meta = metadata.pop()
                new_meta = get_metadata( next_uid, metagraph, subtensor, CLIENT = CLIENT )
                if new_meta == None: continue
                elif new_meta.last_modified != next_meta.last_modified: # The model has updated.
                    metadata.appendleft((next_uid, new_meta)) # Send to the back.
                    continue
                elif config.use_wandb:
                    wandb.log({ "EvalUID": next_uid})
                
                # Load the model.
                try:
                    if 'model' in locals() and model != None:
                        del model
                    torch.cuda.empty_cache()
                    model = download_model( metadata = next_meta, device = 'cpu', CLIENT = CLIENT )
                    model.to(config.device)
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

                    # Forward pass without gradient memory.
                    with torch.no_grad():
                        outputs = model( input_ids=input_ids, labels=labels )
                    losses.append( outputs.loss.item() )
                    
                    # Clear cache to prevent memory leaks
                    del input_ids, labels, outputs
                    torch.cuda.empty_cache()
                    
                # Compute the median loss on all batches from all pages.
                median_loss = np.median(losses)
                
                # Uses an epsilon: (start_block - last_update_block)/temperature reduction.
                # because we run the models in order of upload this puts the oldest model at an advantage.
                # The epsilon decays slowly from config.epsilon until it hits 0 as the temperature term.
                # As time progresses the temperature gets pushed further and further out simulating a slower annealing.
                if config.epsilon_method == 'linear':
                    block_epsilon = config.epsilon * (1 - min(1, max(0, (start_block - last_update_block) / config.temperature)))
                else:
                    # Exponential epsilon with temperature/3 decays quicker and then converges around the temperature term. 
                    block_epsilon = config.epsilon * math.exp(-(start_block - last_update_block) / (config.temperature / 5))
                threshold = best_loss * (1 - block_epsilon) # Threshold based on decay.
                dif = median_loss - threshold
                print (f'BestLoss:{best_loss}, MedianLoss: {median_loss}, Epsilon: {block_epsilon}, Threshold: {threshold}, Dif: {dif}, Temperature: {config.temperature} ')
                if config.use_wandb:
                    wandb.log({ "MedianLoss": median_loss, 'Epsilon': block_epsilon, 'Threshold': threshold, 'BestLoss': best_loss })
                
                # If the average loss is less than the threshold give all incentive to this miner and upload the new state.                          
                if median_loss < threshold:
                    print ('New best loss:', median_loss )
                    best_loss = median_loss
                    
                    # Make the epsilon decay period equivalent to the duration it took to improve the loss.
                    # We use the start_block here rather than subtensor.block since this is when we started evaling the models.
                    # This will increase the temperature if we took a long time to beat the epsilon.
                    config.temperature = (start_block - last_update_block)
                    last_update_block = start_block 
                    
                    # Upload the new best model this is then pulled by all the miners.
                    upload_model(
                        wallet = wallet, 
                        model = model, 
                        extras = { 'sequence_length': config.sequence_length, 'tokenizer_name': config.tokenizer_name }, 
                        bucket = config.bucket,
                        CLIENT = CLIENT
                    )
                    
                    # Set weights on the chain.
                    weights = (1 - config.weights_alpha) * weights + config.weights_alpha * torch.nn.functional.one_hot(torch.tensor(next_uid), num_classes=len(weights)).float()
                    subtensor.set_weights(
                        wallet = wallet,
                        netuid = config.netuid,
                        uids = metagraph.uids,
                        weights = weights,
                        wait_for_inclusion = False,
                        wait_for_finalization = False
                    )  
                    print ( '\tWeights', weights.tolist() )
                    if config.use_wandb:
                        wandb.log({ "BestUID": next_uid, "Temperature": config.temperature })
                        # Log the top weights values
                        top_k = min(5, weights.size(0))
                        top_k_values, top_k_indices = torch.topk(weights, top_k)
                        for i, (value, idx) in enumerate(zip(top_k_values, top_k_indices)):
                            wandb.log({ f"top_weights_{i}": value.item() })
                        
                    # Break the loop here. This gives the earlier miners the advantage.
                    break
                
                                            
        # Handle keyboard interrupts, stops training gracefully.
        except (KeyboardInterrupt, SystemExit):
            to_delete = get_metadata( my_uid, metagraph, subtensor, CLIENT = CLIENT )
            if to_delete != None:
                CLIENT.delete_object( Bucket = config.bucket, Key = to_delete.filename )
            if config.use_wandb:
                wandb.finish()
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
    parser.add_argument('--name', type=str, default='validator', help='Optional name')
    parser.add_argument('--netuid', type=int, default=1, help='Bittensor network uid.')
    parser.add_argument('--bucket', type=str, default='decis', help='S3 bucket name')
    parser.add_argument('--temperature', type=int, default=100, help='Starting epsilon decay range.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Starting epsilon value.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--sequence_length', type=int, default=2048, help='Sequence Length.')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='Tokenizer name.')
    parser.add_argument('--num_pages', type=int, default=1, help='Number of pages to load')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device to use for training')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights and Biases for logging')
    parser.add_argument('--model_type', type=str, choices=['gpt2', 'llama'], default='gpt2', help='Model type to use: gpt2 or llama')
    parser.add_argument('--epsilon_method', type=str, choices=['linear', 'exponential'], default='exponential', help='Epsilon decay method: linear or exponential')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume previous training.')
    parser.add_argument('--weights_alpha', type=float, default=0.1, help='Alpha value for moving average of weights. Higher values preference the current best model.')
    bt.wallet.add_args( parser )
    bt.subtensor.add_args( parser )
    config = bt.config( parser )   
    main( config ) 