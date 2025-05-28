# Visualize-then-Retrieve
<p align="center">
  <a href="https://xiaowu0162.github.io/visret/"><img src="https://img.shields.io/badge/🌐-Website-red" height="23"></a>
  <a href="https://arxiv.org/pdf/2505.20291.pdf"><img src="https://img.shields.io/badge/📝-Paper-blue" height="23"></a>
</p>

Official repository for the paper [Visualized Text-to-Image Retrieval](https://arxiv.org/pdf/2505.20291.pdf).

## Data Release

We release three datasets:
* Visual-RAG: a version of [Visual-RAG](https://github.com/visual-rag/visual-rag) with cleaned image paths. 
* Visual-RAG-ME: a new benchmark annotated for comparing features across related organisms. The benchmark supports both text-to-image retrieval and visual question answering.
* Inquire-Rerank-Hard: a filtered version of [Inquire-Rerank](https://huggingface.co/datasets/evendrow/INQUIRE-Rerank) containing the most challenging questions for off-the-shelf retrievers.

## Code Release

We release the evaluation code for both text-to-image retrieval and visual question answering under each benchmark's folder. Please follow the README there for detailed instructions for data preparation and running the experiments.

## Environment Setup

Our codebase requires running on Nvidia GPUs. To run VisRet, we recommend creating a conda envionment, installing PyTorch, and installing the dependencies in `requirements.txt`.

```
conda create -n visret python=3.9
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
