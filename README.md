# STARLINE: Contrastive Learning with Modality-Aware Graph Refinement for Effective Multimedia Recommendation

This repository provides the official PyTorch implementation of STARLINE, as introduced in the following paper:

> STARLINE: Contrastive Learning with Modality-Aware Graph Refinement for Effective Multimedia Recommendation  
> Taeri Kim, Sohee Ban, Hyunjoon Kim, and Sang-Wook Kim
> In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD'25)


### Requirements
The code has been tested running under Python 3.6.13. The required packages are as follows:
- ```gensim==3.8.3```
- ```pytorch==1.10.2+cu113```
- ```torch_geometric=2.0.3```
- ```sentence_transformers=2.2.0```
- ```pandas```
- ```numpy```
- ```scipy```
- ```tqdm```

### Dataset Preparation
#### Dataset Download
*Baby, Beauty, and Toys & Games*: Download 5-core reviews data, meta data, and image features from [Amazon product dataset](http://jmcauley.ucsd.edu/data/amazon/links.html). Put data into the directory data/{folder}/meta-data/.

*Men Clothing and Women Clothing*: Download Amazon product dataset provided by [MAML](https://github.com/liufancs/MAML). Put data folder into the directory data/.

#### Dataset Preprocessing
Run ```python build_data.py --name={Dataset}```

### Acknowledgement
The structure of this code is largely based on [MONET](https://github.com/Kimyungi/MONET). Thank for their work.
