#!/usr/bin/env bash
python_ver=36
torch_ver=1.1.0
torchvision_ver=0.3.0
device=cpu/
host=https://download.pytorch.org/whl/${device}

wget ${host}torch-${torch_ver}-cp${python_ver}-cp${python_ver}m-linux_x86_64.whl
wget ${host}torchvision-${torchvision_ver}-cp${python_ver}-cp${python_ver}m-linux_x86_64.whl
