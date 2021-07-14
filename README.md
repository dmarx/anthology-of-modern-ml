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
* https://en.wikipedia.org/wiki/List_of_important_publications_in_computer_science#Machine_learning
* https://en.wikipedia.org/wiki/Computational_learning_theory#References

# "Classic" ML

* Lasso/elasticnet
  * 1996 - ["Regression Shrinkage and Selection via the Lasso"](https://statweb.stanford.edu/~tibs/lasso/lasso.pdf) - Robert Tibshirani
  * 2005 - ["Regularization and variable selection via the elastic net"](https://web.stanford.edu/~hastie/Papers/B67.2%20(2005)%20301-320%20Zou%20&%20Hastie.pdf) - Hui Zou, Trevor Hastie
* Boosting
  * 1990 - ["The Strength of Weak Learnability"](http://rob.schapire.net/papers/strengthofweak.pdf) - Robert E. Schapire
* Bagging
  * 1991 - ["Bagging Predictors"](https://www.stat.berkeley.edu/~breiman/bagging.pdf) - Leo Breiman
* random forest
  * 2001 - ["Random Forests"](https://link.springer.com/content/pdf/10.1023/A:1010933404324.pdf) - Leo Breiman
* gradient boosting / Adaboost
* bias-variance tradeoff
* non-parametric bootstrap
  * 1979 - ["Bootstrap Methods: Another Look at the Jackknife"](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full) - Bradley Effron
* permutation testing (target shuffle)
* PCA
  * 1991 - ["Face Recognition Using Eigenfaces"](https://www.cin.ufpe.br/~rps/Artigos/Face%20Recognition%20Using%20Eigenfaces.pdf) - Matthew Turk, Alex Pentland
* ICA
* LSI
* LDA
* SVM
  * 1992 - [A Training Algorithm for Optimal Margin Classifier](https://www.researchgate.net/publication/2376111_A_Training_Algorithm_for_Optimal_Margin_Classifier) - Bernhard E. Boser, Isabelle Guyon, Vladimir N. Vapnik
* NMF
* random projections
* MCMC - metropolis-hastings, HMC, reversible jump, NUTS
* SMOTE
* tSNE
* UMAP
* LSH
  *  2014 - ["LOCALITY PRESERVING HASHING"](https://faculty.ucmerced.edu/mhyang/papers/icip14_lph.pdf) - Yi-Hsuan Tsai, Ming-Hsuan Yang
* Feature hashing / hashing trick 
  * 1989 - ["Fast learning in multi-resolution hierarchies"](https://proceedings.neurips.cc/paper/1988/file/82161242827b703e6acf9c726942a1e4-Paper.pdf) - John Moody
  * 2009 - ["Feature Hashing for Large Scale Multitask Learning"](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf) - Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg
* the kernel trick
* naive bayes
* HMM
* CRF
* RBM - Restricted Boltzman Machine
* GAM - General Additive Models
* MARS - Multivariate adaptive regression splines
  * 1991 - ["Multivariate Adaptive Regression Splines"](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full) - Jerome H. Friedman
* Decision Trees
  * 1986 - (ID3) ["Induction of Decision Trees"](https://link.springer.com/content/pdf/10.1007/BF00116251.pdf) - J. R. Quinlan
  * 1984 - (CART) "Classification and Regression Trees" - in lieu of the book, here's a [topic summary from 2011](http://pages.stat.wisc.edu/~loh/treeprogs/guide/wires11.pdf) -  Breiman L, Friedman JH, Olshen RA, Stone CJ 
* KNN
  * 1967 - ["Nearest Neighbor Pattern Classification"](http://ssg.mit.edu/cal/abs/2000_spring/np_dens/classification/cover67.pdf) - T. M. Cover, P. E. Hart
* Benford's Law
  * 1938 - "The Law of Anomalous Numbers" - Frank Benford
  * 1881 -  "Note on the frequency of use of the different digits in natural numbers" - Simon Newcomb
* Guassian KDE

# Network Graphs / combinatorial optimization

* Dijkstra
* A\*
* Graph anomaly detection (enron)
* Exponential random graphs
* modularity / louvain community detection
  * 2004 - ["Finding community structure in very large networks"](https://arxiv.org/abs/cond-mat/0408187) - Aaron Clauset, M. E. J. Newman, Cristopher Moore
  * 2008 - ["Fast unfolding of communities in large networks"](https://arxiv.org/abs/0803.0476) - Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre
* pagerank
  * 1998 - ["The PageRank Citation Ranking: Bringing Order to the Web"](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) - Larry Page
* knapsack problem
* smallworld
* scale free
* "Networks of Love"
* Genetic algorithms

# Misc optimization and numerical methods

* Newton-raphson
* L-BFGS
* simulated annealing
* FFT
* Constraint Programming / queing theory / OR
  * Uh... here there be dragons. Maybe just leave some breadcrumbs here?

# Neural optimizers

* perceptron algorithm
* SGD / backprop
  * 1986 - ["Learning representations by back-propagating errors"](http://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf) - David Rumelhart, Geoffrey Hinton, Ronald Williams
* Adagrad / RMSProp
  * Probably discussed sufficiently in the Adam paper
* Adam
  * 2014 - ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) - Diederik P. Kingma, Jimmy Ba 
* reverse-mode autodiff
  * see backprop
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
  * 1997 - ["LONG SHORT-TERM MEMORY"](https://www.bioinf.jku.at/publications/older/2604.pdf) - Sepp Hochreiter, Jurgen Schmidhuber
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
* temporal differences
  * 1988 - ["Learning to predict by the methods of temporal differences"](https://link.springer.com/content/pdf/10.1007/BF00115009.pdf) - Richard S. Sutton
* Q learning
  * 1989 - ["Learning from Delayed Rewards"](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) - Christopher Watkins
  * 2013 - (DQN) ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) - Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller


# Hyperparameter tuning

* random search > grid search
  * 2012 - ["Random Search for Hyper-Parameter Optimization"](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) - James Bergstra, Yoshua Bengio
* bayesian / gaussian process (explore/exploit)
* Population based training

# Specific architectures/achievements, and other misc milestones

* alexnet
* BERT
* word2vec
* U-net
* siamese network
  * 2015 - ["Siamese Neural Networks for One-Shot Image Recognition"](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) - Gregory Koch
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
* BNN
  * 1995 - ["Bayesian Methods for Neural Networks"](https://www.microsoft.com/en-us/research/wp-content/uploads/1995/01/NCRG_95_009.pdf) - Christopher Bishop 
* The Netflix Prize
  * 2007 - ["The Netflix Prize"](https://web.archive.org/web/20070927051207/http://www.netflixprize.com/assets/NetflixPrizeKDD_to_appear.pdf) - overview of the contest and dataset
  * 2007 - ["The BellKor solution to the Netflix Prize"](https://www.netflixprize.com/assets/ProgressPrize2007_KorBell.pdf) - Robert M. Bell, Yehuda Koren, Chris Volinsky
  * 2008 - ["The BellKor 2008 Solution to the Netflix Prize"](https://www.netflixprize.com/assets/ProgressPrize2008_BellKor.pdf) - Robert M. Bell, Yehuda Koren, Chris Volinsky
  * 2008 - ["The BigChaos Solution to the Netflix Prize 2008"](https://www.netflixprize.com/assets/ProgressPrize2008_BigChaos.pdf) - Andreas Toscher, Michael Jahrer
* Kaggle Galaxy Zoo

# Learning theory / Deep learning theory / model compression / interpretability

* VC Dimension
  * 1971 - ["On the uniform convergence of relative frequencies of events to their probabilities"](https://courses.engr.illinois.edu/ece544na/fa2014/vapnik71.pdf) - V. Vapnik and A. Chervonenkis
  * 1989 - ["Learnability and the Vapnik-Chervonenkis Dimension "](https://www.trhvidsten.com/docs/classics/Blumer-1989.pdf) - Blumer, A.; Ehrenfeucht, A.; Haussler, D.; Warmuth, M. K. 
* gradient double descent
  * 2019 - ["Deep Double Descent: Where Bigger Models and More Data Hurt"](https://arxiv.org/abs/1912.02292) - Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever
* neural tangent kernel
  * 2018 - ["Neural Tangent Kernel: Convergence and Generalization in Neural Networks"](https://arxiv.org/abs/1806.07572) - Arthur Jacot, Franck Gabriel, Clément Hongler
* lottery ticket hypothesis
  * 2018 - ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](https://arxiv.org/abs/1803.03635) - Jonathan Frankle, Michael Carbin
* manifold hypothesis
* information bottleneck
* generalized degrees of freedom
  * 1998 - "On Measuring and Correcting the Effects of Data Mining and Model Selection" - Jianming Ye
* AIC / BIC
* dropout as ensemblification
  * 2017 - ["Analysis of dropout learning regarded as ensemble learning"](https://arxiv.org/pdf/1706.06859.pdf) - Kazuyuki Hara, Daisuke Saitoh, Hayaru Shouno
* knowledge distillation
  * 2005 - ["Model Compression"](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf) - Cristian Bucila, Rich Caruana, Alexandru Niculescu-Mizil
  * 2015 - ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/pdf/1503.02531.pdf) - Geoffrey Hinton, Oriol Vinyals, Jeff Dean
* model quantization
* SGD = MAP inference
  * 2017 - ["Stochastic Gradient Descent as Approximate Bayesian Inference"](https://arxiv.org/pdf/1704.04289.pdf) - Stephan Mandt, Matthew D. Hoffman, David M. Blei
* shapley scoring
* LIME
* adversarial examples
  * 2014 - ["Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images"](https://arxiv.org/pdf/1412.1897.pdf) - Anh Nguyen, Jason Yosinski, Jeff Clune
* (AU)ROC Curve / PR curve
* Cohen's Kappa / meta-analysis
* PAC learning
  * 1984 - ["A Theory of the Learnable"](http://web.mit.edu/6.435/www/Valiant84.pdf) - L. G. Valiant
* L1/L2 regularization expressable as bayesian priors
  * 1943 - "On the stability of inverse problems" - L2 regularization introduced by Tikhonov, original paper in Russian
* Hilbert spaces
* No Free Lunch
  * 1997 - ["No Free Lunch Theorems for Optimization"](https://ti.arc.nasa.gov/m/profile/dhw/papers/78.pdf) - David H. Wolpert, William G. Macready
* Significance test for the LASSO


# Information theory? 

* Entropy
* Fisher information
* KL divergence


# Causal Modeling / experimentation

* Double machine learning
  * 2016 - ["Double/Debiased Machine Learning for Treatment and Causal Parameters"](https://arxiv.org/abs/1608.00060) - Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins
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
