# export HF_HOME='/path/to/data/huggingface'

cur_dir=$(pwd)

if [ -d "/tmp/llava" ]; then
  echo "Installed, skip."
  exit
else
  echo "Install llava"
  mkdir -p "/tmp/llava"
fi

pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e .
pip3 install -e ".[train]"

pip3 install flash-attn --no-build-isolation