# Anthology of Modern Machine Learning

A curated collection of significant/impactful articles to be treated as a textbook, because sometimes it's just best to go straight to the source. 

I plan to have it organized a couple of ways: 

* broad-brush topics (textbook-ish sections)
* publication date
* parent-child research developments (co-citations?)

This multiple-organization idea might be more amenable to a wiki structure, in which case I could even add paper summaries and abridged versions.

EDIT: Unsurprisingly, others have already done similar stuff. E.g.: https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap . 

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
  * 2014 - ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov

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
  * see also knowledge distillation below
* GAN, DCGAN, WGAN
* Neural ODE
* Neural PDE
* VGG16
* GLoVe
* GLUE task
* inception
* style transfer, content-texture decomposition, weight covariance transfer
* cyclegan/discogan
* autoencoders
* VAE
* Amortized VAE
* YOLO
* GPT-2
* ULM
* seq2seq
* pix2pix
* fasttext
* wordpiece tokenization
  * 2016 - ["Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"](https://arxiv.org/abs/1609.08144v2) - Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean
* GPT-3
  * 2020 - ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) - Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

# Learning theory / Deep learning theory / model compression

* gradient double descent
  * 2019 - ["Deep Double Descent: Where Bigger Models and More Data Hurt"](https://arxiv.org/abs/1912.02292) - Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever
* neural tangent kernel
  * 2018 - ["Neural Tangent Kernel: Convergence and Generalization in Neural Networks"](https://arxiv.org/abs/1806.07572) - Arthur Jacot, Franck Gabriel, Clément Hongler
* lottery ticket hypothesis
  * 2018 - ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](https://arxiv.org/abs/1803.03635) - Jonathan Frankle, Michael Carbin
* manifold hypothesis
* information bottleneck
* generalized degrees of freedom
* AIC
* dropout as ensemblification
  * 2017 - ["Analysis of dropout learning regarded as ensemble learning"](https://arxiv.org/pdf/1706.06859.pdf) - Kazuyuki Hara, Daisuke Saitoh, Hayaru Shouno
* knowledge distillation
  * 2005 - ["Model Compression"](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf) - Cristian Bucila, Rich Caruana, Alexandru Niculescu-Mizil
  * 2015 - ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/pdf/1503.02531.pdf) - Geoffrey Hinton, Oriol Vinyals, Jeff Dean
* model quantization
* SGD = MAP inference
