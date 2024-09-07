pm2 delete all

# Super hacky way of setting these vars.
sed -i 's/Self::set_target_registrations_per_interval(netuid, 1);/Self::set_target_registrations_per_interval(netuid, 10);/' subtensor/pallets/subtensor/src/coinbase/root.rs
# Parse command line arguments for --bucket, --use-wandb, and --model_type
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --bucket) bucket="$2"; shift ;;
        --use-wandb) use_wandb=true ;;
        --model_type) model_type="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Set default values if not provided

# Set default values if not provided
model_type=${model_type:-'gpt2'}
bucket=${bucket:-'decis'}
use_wandb=${use_wandb:-false}

echo "Model Type: $model_type"
echo "Using bucket: $bucket"
echo "Use Weights and Biases: $use_wandb"

# Run the local subtensor.
if ! pm2 list | grep -q -- "localchain"; then
    pm2 start subtensor/scripts/localnet.sh --name localchain
else
    pm2 delete localchain
    pm2 start subtensor/scripts/localnet.sh --name localchain
fi

# Check if pm2 processes for Alice, Charlie, and Bob exist and delete them if they do
for process in Alice Charlie Bob; do
    if pm2 list | grep -q -- "$process"; then
        pm2 delete "$process"
    fi
done

while true; do
    output=$(btcli s list --subtensor.chain_endpoint ws://127.0.0.1:9946 2>&1)
    if echo "$output" | grep -q "Could not connect to local network with ws://127.0.0.1:9946 chain endpoint. Exiting..."; then
        echo "Waiting for local chain to spin up..."
        sleep 5
    else
        echo "$output"
        break
    fi
done

# Create Alice, Bob, Charlie, Dave, Eve, Ferdie
echo "Creating wallets for Alice, Bob, Charlie, Dave, Eve, and Ferdie ..."
python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Alice'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_new_hotkey( overwrite=True, use_password = False, suppress = False); print(w)"
python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Bob'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_new_hotkey( overwrite=True, use_password = False, suppress = False); print(w)"
python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Charlie'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_new_hotkey( overwrite=True, use_password = False, suppress = False); print(w)"
python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Dave'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_new_hotkey( overwrite=True, use_password = False, suppress = False); print(w)"
python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Eve'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_new_hotkey( overwrite=True, use_password = False, suppress = False); print(w)"
python3 -c "import bittensor as bt; w = bt.wallet( name = 'Alice', hotkey = 'Ferdie'); w.create_coldkey_from_uri('//Alice', overwrite=True, use_password = False, suppress = True); w.create_new_hotkey( overwrite=True, use_password = False, suppress = False); print(w)"

echo "Creating subnet 1 ... "
btcli s create --wallet.name Alice --wallet.hotkey Alice --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Setting Hparams ..."
btcli sudo set --wallet.name Alice --netuid 1 --param 'min_allowed_weights' --value 0 --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli sudo set --wallet.name Alice --netuid 1 --param 'max_weight_limit' --value 65535 --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Registering Keys to subnet 1 ... "
btcli s register --netuid 1 --wallet.name Alice --wallet.hotkey Alice --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli s register --netuid 1 --wallet.name Alice --wallet.hotkey Bob --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli s register --netuid 1 --wallet.name Alice --wallet.hotkey Charlie --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli s register --netuid 1 --wallet.name Alice --wallet.hotkey Dave --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli s register --netuid 1 --wallet.name Alice --wallet.hotkey Eve --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
btcli s register --netuid 1 --wallet.name Alice --wallet.hotkey Ferdie --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946

echo "Adding Stake ..."
btcli stake add --amount 10000 --wallet.name Alice --wallet.hotkey Alice --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli stake add --amount 1000 --wallet.name Bob --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli stake add --amount 100 --wallet.name Charlie --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli stake add --all --wallet.name Dave --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli stake add --all --wallet.name Eve --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli stake add --all --wallet.name Ferdie --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946

# echo "Registering Keys to root 0 ..."
# btcli root register --netuid 0 --wallet.name Alice --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli root register --netuid 0 --wallet.name Bob --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli root register --netuid 0 --wallet.name Charlie --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946

# echo "Setting root weights ... "
# btcli root weights --netuids 1 --weights 1 --wallet.name Alice --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli root weights --netuids 1 --weights 1 --wallet.name Bob --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
# btcli root weights --netuids 1 --weights 1 --wallet.name Charlie --wallet.hotkey default --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946

btcli s metagraph --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946

# Start all.
if [ "$use_wandb" = true ]; then
    wandb login
    pm2 start validator.py --interpreter python3 --name Alice -- --wallet.name Alice --wallet.hotkey Alice --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:0 --bucket $bucket --use_wandb --model_type $model_type
    pm2 start miner.py --interpreter python3 --name Bob -- --wallet.name Alice --wallet.hotkey Bob --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:1 --bucket $bucket --use_wandb --learning_rate 0.00001
    pm2 start miner.py --interpreter python3 --name Charlie -- --wallet.name Alice --wallet.hotkey Charlie --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:2 --bucket $bucket --use_wandb --learning_rate 0.00005
    pm2 start miner.py --interpreter python3 --name Dave -- --wallet.name Alice --wallet.hotkey Dave --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:3 --bucket $bucket --use_wandb --learning_rate 0.0001
    pm2 start miner.py --interpreter python3 --name Eve -- --wallet.name Alice --wallet.hotkey Eve --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:4 --bucket $bucket --use_wandb --learning_rate 0.0005
    pm2 start miner.py --interpreter python3 --name Ferdie -- --wallet.name Alice --wallet.hotkey Ferdie --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:5 --bucket $bucket --use_wandb --learning_rate 0.001
else
    pm2 start validator.py --interpreter python3 --name Alice -- --wallet.name Alice --wallet.hotkey Alice --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:0 --bucket $bucket --model_type $model_type
    pm2 start miner.py --interpreter python3 --name Bob -- --wallet.name Alice --wallet.hotkey Bob  --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:1 --bucket $bucket
    pm2 start miner.py --interpreter python3 --name Charlie -- --wallet.name Alice --wallet.hotkey Charlie  --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:2 --bucket $bucket
    pm2 start miner.py --interpreter python3 --name Dave -- --wallet.name Alice --wallet.hotkey Dave  --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:3 --bucket $bucket
    pm2 start miner.py --interpreter python3 --name Eve -- --wallet.name Alice --wallet.hotkey Eve  --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:4 --bucket $bucket
    pm2 start miner.py --interpreter python3 --name Ferdie -- --wallet.name Alice --wallet.hotkey Ferdie --subtensor.chain_endpoint ws://127.0.0.1:9946 --device cuda:5 --bucket $bucket
fi

# Log the validator
pm2 logs Alice
