<div style="text-align: center;">
  <h1>/\ |_ /\ |) |) | |\|</h1>
</div>


# Installing S3
Aladdin requires an AWS S3 bucket for internode communication. Follow these steps to set one up and set your environment vars. 
1. **Create an AWS Account**:
   - If you don't already have an AWS account, go to [AWS](https://aws.amazon.com/) and create one.

2. **Create an S3 Bucket**:
   - Open the AWS Management Console.
   - Navigate to the S3 service.
   - Click on "Create bucket".
   - Follow the prompts to set up your bucket.

3. **Create an IAM User**:
   - Navigate to the IAM (Identity and Access Management) service.
   - Click on "Users" and then "Add user".
   - Provide a username and select "Programmatic access".
   - Click "Next: Permissions".
   - Attach the "AmazonS3FullAccess" policy to the user.
   - Click through the remaining steps and create the user.
   - Save the Access Key ID and Secret Access Key provided at the end.

4. **Set Up Environment Variables**:
   - Open your terminal.
   - Run the following commands to set your AWS credentials as environment variables:
     ```bash
     export AWS_ACCESS_KEY_ID=your_access_key_id
     export AWS_SECRET_ACCESS_KEY=your_secret_access_key
     ```
   - Replace `your_access_key_id` and `your_secret_access_key` with the values you saved from the IAM user creation.

5. **Verify Access**:
   - You can verify your access by running the following command:
     ```bash
     aws s3 ls
     ```
   - This should list all your S3 buckets if the credentials are set up correctly.



# Installing Subtensor:
The demo makes use of a local subtensor chain. Run these commands to install Subtensor on your linux box.
```
git clone git@github.com:opentensor/subtensor.git
sudo apt update
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
sudo apt install -y git clang curl libssl-dev llvm libudev-dev
sudo apt-get install protobuf-compiler
rustup target add wasm32-unknown-unknown --toolchain stable-x86_64-unknown-linux-gnu
```

# Installing Python:
Install the python dependencies.
```bash
# Create a virtual environment with Python 3.11
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

# Running:
You can use the start script to pop up a network running the demo. The script does the following
1. Deletes previous running processes from this script.
2. Creates a local subtensor chain.
3. Registers subnet 1.
4. Stakes to accounts Alice, Bob, Charlie
5. Registers Alice, Bob and Charline on subnet 1 and root.
6. Starts Alice as a validator with 1000 TAO staked on Subnet 1.
7. Starts Bob and Charlie as miners on Subnet 1.
8. Optionally starts wandb run.
```bash
./start.sh --bucket <your S3 bucket> --use-wandb
```

# Running additional Miners.
Alternatively you can register your own miners on the subnet. (see start.sh for how to create one).
```bash
btcli faucet (I cant remember the command) --wallet.name <your coldkey> --wallet.hotkey <your hotkey>
btcli subnet register --netuid 1 --wallet.name <your coldkey> --wallet.hotkey <your hotkey> --no_prompt --subtensor.chain_endpoint ws://127.0.0.1:9946
python3 miner.py --netuid 1 --wallet.name <your coldkey> --wallet.hotkey <your hotkey> --device <your device> --subtensor.chain_endpoint ws://127.0.0.1:9946 
```

