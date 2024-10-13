# NN-Project---Matformer
Gavriel Di Nepi 2067753
## Introduction
In this repository, is presented a PyTorch re-implementation of the paper "MATFORMER: NESTED TRANSFORMER FOR ELASTIC INFERENCE." The paper introduces MatFormer, a nested Transformer architecture designed to provide elasticity under various deployment constraints. Since Transformer models are deployed in a wide range of environments, models of different sizes are often required. With MatFormer, it is possible to train a single model and extract multiple smaller models from it, tailored to different resource limitations.

Link to the article: (https://arxiv.org/abs/2310.07707)

## Architecture
MatLM consists of ***l  layers*** of MatFormer Blocks. The Feed Forward Network (FFN) within each block follows a Matryoshka structure, where the maximum number of neurons corresponds to d_ffn, and the network is divided into 4 levels of granularity. Neurons from the smaller granularities are shared with the larger ones. This sharing of neurons, combined with the joint loss computation across different granularities, encourages the model to utilize the first m_i neurons in a more significant way.


For the rest of the network I used the normal decoder structure:
 - Embeddings
 - Positional Embeddings 
 - l layers of MatFormer Block
 - Normalization layer 
 - Linear layer
 
## How to run the code
To run the code, I recommend downloading the notebook and the weights of all models. Place the weights in a folder named "weights." After that, simply press "Run All" to see the results.



