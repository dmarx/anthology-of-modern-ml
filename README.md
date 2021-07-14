# Anthology of Modern Machine Learning

A curated collection of significant/impactful articles to be treated as a textbook. Plan to have it organized a couple of ways: 

* broad-brush topics (textbook-ish sections)
* publication date
* parent-child research developments (co-citations?)


# "Classic" ML

* Lasso/elasticnet
  * 1996 - ["Regression Shrinkage and Selection via the Lasso"](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf) - Robert Tibshirani
  * 2005 - ["Regularization and variable selection via the elastic net"](https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf) - Hui Zou, Trevor Hastie
* random forest
  * 2001 - ["Random Forests"](https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf) - Leo Breiman
* gradient boosting / Adaboost
* bias-variance tradeoff
* non-parametric bootstrap
* permutation testing (target shuffle)
* PCA
* ICA
* LSI
* LDA
* SVM
* NMF
* random projections
* MCMC - metropolis-hastings, HMC, reversible jump, NUTS
* SMOTE
* tSNE
* UMAP


# Network Graphs / combinatorial optimization

* Dijkstra
* A\*
* Graph anomaly detection (enron)
* Exponential random graphs
* louvain community detection
* pagerank
* knapsack problem
* smallworld
* scale free
* "Networks of Love"

# Misc optimization

* Newton-raphson
* L-BFGS
* simulated annealing

# Neural optimizers

* perceptron algorithm
* SGD / backprop
  * 1986 - ["Learning representations by back-propagating errors"](http://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf) - David Rumelhart, Geoffrey Hinton, Ronald Williams
* Adam
* Adagrad
* reverse-mode autodiff
* gradient clipping
* learning rate scheduling

# Neural activations

* sigmoid
* tanh
* ReLU
* leaky Relu

# Neural layers

* MLP
* convolutions (+ pooling)
* dilated convolutions (Wavenet)
* LSTM
* GRU
* Residual connections - Resnets + highway networks
* batchnorm
* attention
* self-attention -> transformers
  * 2017 - ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
* dropout

# RL

* multi-armed bandit
* Q learning

# Hyperparameter tuning

* grid search
* random search > grid search
* bayesian / gaussian process (explore/exploit)
* Population based training

# Specific architectures/achievements

* alexnet
* BERT
* word2vec
* U-net
* siamese network
* student-teacher transfer learning, catastrophic forgetting
* GAN, DCGAN, WGAN
* Neural ODE
* Neural PDE
* VGG16
* GLoVe
* GLUE task
* inception
* style transfer, content-texture decomposition, weight covariance transfer
* cyclegan/discogan

# Learning theory / Deep learning theory / model compression

* gradient double descent
  * 2019 - ["Deep Double Descent: Where Bigger Models and More Data Hurt"](https://arxiv.org/abs/1912.02292) - Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever
* neural tangent kernel
  * 2018 - ["Neural Tangent Kernel: Convergence and Generalization in Neural Networks"](https://arxiv.org/abs/1806.07572) - Arthur Jacot, Franck Gabriel, Clément Hongler
* lottery ticket hypothesis
* manifold hypothesis
* information bottleneck
* generalized degrees of freedom
* AIC
* dropout as ensemblification
* hinton's dark knowledge
* model quantization
