#!/bin/bash

echo "Creating the datasets directory" 
mkdir -p datasets/fairface && cd datasets/fairface 
echo "Downloading the fairface dataset ... "

gdown 1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86 # From the fairface repository
gdown 1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH
gdown 1wOdja-ezstMEp81tX1a-EYkFebev4h7D

unzip fairface-img-margin025-trainval.zip

export LP_FAIRFACE_PATH=$(pwd)/datasets/fairface
