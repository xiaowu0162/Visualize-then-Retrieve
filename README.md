# visualize-then-retrieve
Official repository for the paper Visualized Text-to-Image Retrieval


## Data Release

We release annotations and 


## Environment Setup

Our codebase requires running on Nvidia GPUs. To run VisRet, we recommend creating a conda envionment, installing PyTorch, and installing the dependencies in `requirements.txt`.

```
conda create -n visret python=3.9
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
