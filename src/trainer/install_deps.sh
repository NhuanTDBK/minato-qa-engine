sudo apt update
sudo apt install python3-pip
pip3 install -U pip uv
export PATH=$PATH:/home/ubuntu/.local/bin
uv pip install -U  accelerate torch==2.5.0 torchvision datasets evaluate  "numpy<2" qwen-vl-utils deepspeed==0.15.4 trl==0.14.0 liger-kernel  unsloth rouge_score 
mkdir training