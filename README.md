# Anthology of Modern Machine Learning

A curated collection of significant/impactful articles to be treated as a textbook, because sometimes it's just best to go straight to the source. 

I plan to have it organized a couple of ways: 

* broad-brush topics (textbook-ish sections)
* publication date
* parent-child research developments (co-citations?)

This multiple-organization idea might be more amenable to a wiki structure, in which case I could even add paper summaries and abridged versions.

Some similar projects worth checking out: 

* https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap
* https://github.com/terryum/awesome-deep-learning-papers

# "Classic" ML

* Lasso/elasticnet
  * 1996 - ["Regression Shrinkage and Selection via the Lasso"](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf) - Robert Tibshirani
  * 2005 - ["Regularization and variable selection via the elastic net"](https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf) - Hui Zou, Trevor Hastie
* random forest
  * 2001 - ["Random Forests"](https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf) - Leo Breiman
* gradient boosting / Adaboost
* bias-variance tradeoff
* non-parametric bootstrap
  * 1979 - ["Bootstrap Methods: Another Look at the Jackknife"](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full) - Bradley Effron
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

# Misc optimization and numerical methods

* Newton-raphson
* L-BFGS
* simulated annealing
* FFT

# Neural optimizers

* perceptron algorithm
* SGD / backprop
  * 1986 - ["Learning representations by back-propagating errors"](http://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf) - David Rumelhart, Geoffrey Hinton, Ronald Williams
* Adagrad / RMSProp
  * Probably discussed sufficiently in the Adam paper
* Adam
  * 2014 - ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) - Diederik P. Kingma, Jimmy Ba 
* reverse-mode autodiff
* gradient clipping
* learning rate scheduling
* distributed training
* federated learning

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
  * 1989 - ["Learning from Delayed Rewards"](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) - Christopher Watkins
  * 2013 - (DQN) ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) - Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller


# Hyperparameter tuning

* random search > grid search
  * 2012 - ["Random Search for Hyper-Parameter Optimization"](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) - James Bergstra, Yoshua Bengio
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
* wordpiece tokenization / BPE
  * 2015 - ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) - Rico Sennrich, Barry Haddow, Alexandra Birch
  * 2016 - ["Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"](https://arxiv.org/abs/1609.08144v2) - Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean
* GPT-3
  * 2020 - ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) - Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei

# Learning theory / Deep learning theory / model compression / interpretability

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
* shapley scoring
* LIME
* adversarial examples
  * 2014 - ["Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images"](https://arxiv.org/pdf/1412.1897.pdf) - Anh Nguyen, Jason Yosinski, Jeff Clune

# Causal Modeling / experimentation

* Double machine learning
* Doubly robust inference
* Pearl's do calculus and graphical modeling
* Rubin's potential outcomes model
* model identification
* d-separation
* propensity scoring/matching
* item-response model and adaptive testing
* bandit learning for on-line experimentation

# Time series forecasting

* ARMA / ARIMA / ARIMAX
* sin/cos cyclical day encodings
* RNN forecasting
  * 1991 - ["Recurrent Networks and NARMA Modeling"](https://proceedings.neurips.cc/paper/1991/file/5ef0b4eba35ab2d6180b0bca7e46b6f9-Paper.pdf) - J. Connor, L. Atlas, R. Martin
  * 2017 - ["A Multi-Horizon Quantile Recurrent Forecaster"](https://arxiv.org/pdf/1711.11053.pdf) - (Amazon) Ruofeng Wen, Kari Torkkola, Balakrishnan Narayanaswamy, Dhruv Madeka
* FB Prophet / bayesian
