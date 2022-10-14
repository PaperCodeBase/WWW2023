# WWW2023

This repository is the official implementation of our paper: *Cross-Modality Mutual Learning for Enhancing Smart Contract Vulnerability Detection on Bytecode*. 


## Requirements

- Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

- Required Packages

Run the following commands to install the required packages.
```shell
pip install torch==1.0.0
pip install numpy==1.15.4
pip install six==1.15.0
pip install scikit-learn==0.24.1
pip install scipy==1.1.0
```


- Datasets

We notice that there is still a lack of benchmark datasets for smart contract vulnerability detection on bytecode. Indeed, it is non-trivial to obtain a high-quality dataset that contains labeled vulnerable functions of smart contracts, attributing to the demand for qualified expertise. Motivated by this, we construct and release a benchmark dataset, which concerns four types of vulnerabilities, namely reentrancy, timestamp, integer overflow/underflow, and delegatecall.

The dataset has been created by collecting smart contracts from three different sources: 1) Ethereum platform (more than 96%), 2) GitHub repositories, and 3) blog posts that analyze contracts.

More details for the dataset guidance can be found on our dataset page at [Dataset](https://github.com/PaperCodeBase/Smart-Contract-Dataset).



- SourceGraphExtractor

For source code, we use a code semantic graph to frame the control- and data- dependencies in the code. We extract two kinds of graph nodes and three types of temporal edges. We refer to the open-sourced tool of [GraphExtractor](https://github.com/Messi-Q/GNNSCVulDetector).



- BinaryCFGExtractor

For bytecode, we develop the automated tool (i.e., BinaryCFGExtractor) to extract the control flow graph in the code, which consists of bytecode blocks (i.e., nodes) and control flow edges. We released BinaryCFGExtractor on our tool page at [BinaryCFGExtractor](https://github.com/PaperCodeBase/BinaryCFGExtractor).


- GraphFeatureExtractor

After obtaining the two kinds of graphs, we build upon the architecture of a graph neural network to learn the high-level graph semantic embeddings of both the source code and the bytecode. Specifically, we released the GraphFeatureExtractor on our tool page at [GraphFeatureExtractor](https://github.com/PaperCodeBase/GraphFeatureExtractor).



- BertPretainFinetune

For processing the bytecode blocks in a CFG, we propose a pre-trained model BERT to handle it. First, we pre-train the BERT through an instruction-level task and a block-level task. Second, considering the discrepancy between different vulnerabilities, we enforce a dedicated fine-tuning of the pre-trained BERT on each of the four vulnerabilities. Finally, the features of bytecode blocks are extracted by the fine-tuned BERT.

More details for the pre-training and fine-tuning of the BERT can be found on our tool page at [BertPretrainFinetune](https://github.com/PaperCodeBase/BertPretrainFinetune).





## Model Training 

- Training

To train a dual-modality teacher network (auxiliary network) and a single-modality student network (main network), we can run the following command:

```train
python train_new.py --dataset timestamp --batch_size 32 --lr 1e-3 --wd 1e-4 --epoch 20 --shuffle True --cuda True 
```





