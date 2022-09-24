#!/bin/zsh

# Credits:
# original post by https://lazycoder.ro/posts/using-gpt-neo-on-m1-mac/

# install tensorflow deps + tensorflow + metal plugin
conda install -c apple tensorflow-deps
pip install tensorflow-macos
pip install tensorflow-metal

# install jupyter, pandas and whatnot
conda install -c conda-forge -y pandas jupyter

# install rust toolkit - if you don't have it
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# install tokenizers
mkdir -p Projects/lab/tfsetup & cd Projects/lab/tfsetup
git clone https://github.com/huggingface/tokenizers
cd tokenizers/bindings/python

# compile tokenizers - should be pretty fast on your m1
pip install setuptools_rust
python setup.py install

# install transformers using pip
pip install git+https://github.com/huggingface/transformers

# prerequisites to build happytransformer
arch -arm64 brew install cmake
arch -arm64 brew install pkgconfig

# building happytransformer
pip install happytransformer

# Then refer to this code and play around 

'''
from happytransformer import HappyGeneration

gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-125M")

input = "What is the stock market?"
result = gen.generate_text(input)

print(result.text)

'''

