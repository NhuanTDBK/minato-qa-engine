sudo apt update
sudo apt install -y python3-pip
pip3 install -U pip uv
export PATH=$PATH:/home/ubuntu/.local/bin
uv pip install --system -U accelerate datasets evaluate unsloth "numpy<2" qwen-vl-utils deepspeed==0.15.4 trl==0.14.0 liger_kernel transformers rouge_score
mkdir training