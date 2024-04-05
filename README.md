# Anthology of Modern Machine Learning

A curated collection of significant/impactful articles to be treated as a textbook, because sometimes it's just best to go straight to the source. My hope is to provide a reference for understanding important developments in the historical context that motivated them, e.g. the problems the authors were attempting to solve, what particular features of the discovery were considered especially novel or impressive when it was first published, what the competing theories or techniques at the time were, etc.

Someday this will be organized better.

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
* Adaboost
  * 1997 - (AdaBoost) ["A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting"](https://www.sciencedirect.com/science/article/pii/S002200009791504X) - Yoav Freund, Robert E Schapire
  * 1999 - ["Improved Boosting Algorithms Using Confidence-rated Predictions"](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1999-ML-Improved%20boosting%20algorithms%20using%20confidence-rated%20predictions%20(Schapire%20y%20Singer\\).pdf) - Yoav Freund, Robert E Schapire
* gradient boosting
  * 2016 - ["XGBoost: A Scalable Tree Boosting System"](https://arxiv.org/pdf/1603.02754.pdf) - Tianqi Chen, Carlos Guestrin
* bias-variance tradeoff
  * 1997 - ["Bias Plus Variance Decomposition for Zero-One Loss Functions"](https://www.researchgate.net/publication/2793430_Bias_Plus_Variance_Decomposition_for_Zero-One_Loss_Functions) - Ron Kohavi, David H. Wolpert
* non-parametric bootstrap
  * 1979 - ["Bootstrap Methods: Another Look at the Jackknife"](https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.full) - Bradley Effron
* permutation testing (target shuffle)
* PCA
  * 1991 - ["Face Recognition Using Eigenfaces"](https://www.cin.ufpe.br/~rps/Artigos/Face%20Recognition%20Using%20Eigenfaces.pdf) - Matthew Turk, Alex Pentland
* nonlinear PCA variants
  * 1989 - ["Principal Curves"](https://web.stanford.edu/~hastie/Papers/Principal_Curves.pdf) - Trevor Hastie, Werner Stuetzle
* ICA
* LSI/LSA
  * 1988 - ["Indexing by Latent Semantic Analysis"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.62.1152&rep=rep1&type=pdf) - Scott Deerwester, Susan T. Dumais, George W. Furnas, Thomas K. Landauer, Richard Harshman
* LDA
  * 2003 - ["Latent Dirichlet Allocation"](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) - David M. Blei, Andrew Ng, Michael I. Jordan
* SVM
  * 1992 - [A Training Algorithm for Optimal Margin Classifier](https://www.researchgate.net/publication/2376111_A_Training_Algorithm_for_Optimal_Margin_Classifier) - Bernhard E. Boser, Isabelle Guyon, Vladimir N. Vapnik
* NMF
  * 2003 - ["Document Clustering Based On Non-negative Matrix Factorization"](https://people.eecs.berkeley.edu/~jfc/hcc/courseSP05/lecs/lec14/NMF03.pdf) - Wei Xu, Xin Liu, Yihong Gon
* random projections
  * https://cseweb.ucsd.edu/~dasgupta/papers/randomf.pdf
  * See also Johnson-Lindenstrauss lemma
* MCMC - metropolis-hastings, HMC, reversible jump, NUTS
* SMOTE
  * 2002 - ["Smote: synthetic minority over-sampling technique"](https://www.jair.org/index.php/jair/article/view/10302/24590) - Nitesh V Chawla, Kevin W Bowyer, Lawrence O Hall, W Philip Kegelmeyer
* tSNE
  * 2002 - ["Stochastic Neighbor Embedding"](https://cs.nyu.edu/~roweis/papers/sne_final.pdf) - Geoffrey Hinton, Sam Roweis 
  * 2008 - ["Visualizing Data using t-SNE"](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) - Laurens van der Maaten, Geoffrey Hinton
* UMAP
  * 2018 - ["UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction"](https://arxiv.org/abs/1802.03426) - Leland McInnes, John Healy, James Melville
* LSH
  *  2014 - ["LOCALITY PRESERVING HASHING"](https://faculty.ucmerced.edu/mhyang/papers/icip14_lph.pdf) - Yi-Hsuan Tsai, Ming-Hsuan Yang
* Feature hashing / hashing trick 
  * 1989 - ["Fast learning in multi-resolution hierarchies"](https://proceedings.neurips.cc/paper/1988/file/82161242827b703e6acf9c726942a1e4-Paper.pdf) - John Moody
  * 2009 - ["Feature Hashing for Large Scale Multitask Learning"](http://alex.smola.org/papers/2009/Weinbergeretal09.pdf) - Kilian Weinberger; Anirban Dasgupta; John Langford; Alex Smola; Josh Attenberg
* the kernel trick
* naive bayes
* ~~HMM~~
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
* Boruta
  * 2010 - ["Feature Selection with the Boruta Package"](https://www.jstatsoft.org/article/view/v036i11) - Miron B. Kursa, Witold R. Rudnicki
* Step-wise regression / forward selection / backwards elimination / recursive feature elimination
* kalman filter
  * 1960 - ["A New Approach to Linear Filtering and Prediction Problems"](http://cs-www.cs.yale.edu/homes/yry/readings/general/Kalman1960.pdf) - R. E. Kalman
* restricted boltzman machine
* Deep belief networks
* Scree plot
  * 1966 - "The Scree Test For The Number Of Factors" - Raymond B. Cattell
* Collaborative Filtering (SVD and otherwise)
* Market basket analysis
* Process mining
* self-organizing maps
  * 1982 - ["Self-Organized Formation of Topologically Correct Feature Maps"](https://tcosmo.github.io/assets/soms/doc/kohonen1982.pdf) - Teuvo Kohonen
* Good overview of modeling process
  * 2020 - ["Bayesian workflow"](https://arxiv.org/pdf/2011.01808v1.pdf) - Andrew Gelman, Aki Vehtari, Daniel Simpson, Charles C. Margossian, Bob Carpenter, Yuling Yao, Lauren Kennedy, Jonah Gabry, Paul-Christian Bürkner, Martin Modrák

* poisson bootstrap
  * 2012 - [ESTIMATING UNCERTAINTY FOR MASSIVE DATA STREAMS](https://research.google/pubs/pub43157/) - Nicholas Chamandy, Omkar Muralidharan,
Amir Najmi, Siddartha Naidu 

* constrained optimization / Linear Programming
  * 1947 - Simplex Algorithm (?) - George Dantzig

* compressed sensing
  * 2004 - ["Compressed Sensing"](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Donoho04.pdf) - David L. Donoho

# Network Graphs / combinatorial optimization

* Graph anomaly detection (enron)
  * 2005 - ["Scan Statistics On Enron Graphs"](https://www.researchgate.net/publication/220556790_Scan_Statistics_on_Enron_Graphs) - Carey E. Priebe, John M. Conroy, David J. Marchette, Youngser Park
* Exponential random graphs
* modularity / louvain community detection
  * 2004 - ["Finding community structure in very large networks"](https://arxiv.org/abs/cond-mat/0408187) - Aaron Clauset, M. E. J. Newman, Cristopher Moore
  * 2008 - ["Fast unfolding of communities in large networks"](https://arxiv.org/abs/0803.0476) - Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre
* pagerank
  * 1998 - ["The PageRank Citation Ranking: Bringing Order to the Web"](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf) - Larry Page
* smallworld
* scale free
* "Networks of Love"
  * 2004 - [Chains of Affection: The Structure of Adolescent Romantic and Sexual Networks](https://people.duke.edu/~jmoody77/chains.pdf) - Peter S. Bearman, James Moody, Katherine Stovel
* Genetic algorithms
* force-directed/spring layout (fruchterman-reingold I think?)
* label propagation
  * 2002 - ["Learning From Labeled and Unlabeled Data With Label Propagation"](http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf) - Xiaojin Zhu, Zoubin Ghahramani

# Geometric Deep Learning and ML applications of group theory/representation theory

* Foundational Books
  * 2008 - [Group theoretical methods in machine learning (thesis)](https://people.cs.uchicago.edu/~risi/papers/KondorThesis.pdf) - Risi Kondor
  * 2021 - [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) - Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković

# Misc optimization and numerical methods

* Expectation maximization
* Newton-raphson
* L-BFGS
* simulated annealing
* FFT
  * 1965 - [An Algorithm for the Machine Calculation of Complex Fourier Series](https://www.ams.org/journals/mcom/1965-19-090/S0025-5718-1965-0178586-1/S0025-5718-1965-0178586-1.pdf) - James Cooley, John Tukey
* Constraint Programming / queing theory / OR
  * Uh... here there be dragons. Maybe just leave some breadcrumbs here?

# Neural optimizers

* perceptron algorithm
* SGD / backprop
  * 1986 - ["Learning representations by back-propagating errors"](http://www.cs.utoronto.ca/~hinton/absps/naturebp.pdf) - David Rumelhart, Geoffrey Hinton, Ronald Williams
  * 1998 - ["Efficient Backprop"](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) -  Yann LeCun, Leon Bottou, Genevieve B. Orr and Klaus-Robert Müller
* Adagrad / RMSProp
  * Probably discussed sufficiently in the Adam paper
* Adam
  * 2014 - ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) - Diederik P. Kingma, Jimmy Ba 
* reverse-mode autodiff
  * see backprop
* gradient clipping
* learning rate scheduling
* distributed training
  * 2011 - ["HOGWILD!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent"](https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf) - Feng Niu, Benjamin Recht, Christopher Re, Stephen J. Wright

* ZeRo Offload, Zero Redundandancy Optimizers
  * 2019 - ["ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"](https://arxiv.org/abs/1910.02054) - Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He

* federated learning

* K-FAC for approximating Fisher Information, Hessian
  * 2015 - ["Optimizing Neural Networks with Kronecker-factored Approximate Curvature"](https://arxiv.org/abs/1503.05671) - James Martens, Roger Grosse

# Neural activations

* sigmoid
* ReLU
  * 2011 - ["Deep Sparse Rectifier Neural Networks"](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf) - Xavier Glorot,  Antoine Bordes, Yoshua Bengio
  * See also AlexNet
* GELU
  * 2016 - ["Gaussian Error Linear Units (GELUs)"](https://arxiv.org/pdf/1606.08415v4.pdf) -  Dan Hendrycks, Kevin Gimpel 
* Gumbel quantization
  * 2016 - ["Categorical Reparameterization with Gumbel-Softmax"](https://arxiv.org/abs/1611.01144v5) - Eric Jang, Shixiang Gu, Ben Poole

# Neural initializations

* Xavier/Glorot initialization - vanishing/exploding gradients
  * 2010 - ["Understanding the difficulty of training deep feedforward neural networks"](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) - Xavier Glorot, Yoshua Bengio

# Neural layers

* MLP
* convolutions (+ pooling)
* depthwise-seperable convolutions (mobilenet?)
* dilated convolutions (Wavenet)
* squeeze-and-excitation block
  * 2017 - ["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507) - Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu 
* LSTM
  * 1997 - ["LONG SHORT-TERM MEMORY"](https://www.bioinf.jku.at/publications/older/2604.pdf) - Sepp Hochreiter, Jurgen Schmidhuber
* Residual connections - Resnets + highway networks
  * 2015 - ["Highway Networks"](https://arxiv.org/abs/1505.00387) - Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
  * 2015 - ["Deep Residual Learning for Image Recognition"](https://arxiv.org/pdf/1512.03385.pdf) - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
* batchnorm
  * 2015 - ["Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"](https://arxiv.org/abs/1502.03167) - Sergey Ioffe, Christian Szegedy
* additive attention
  * see also Alex Graves 2013
  * 2014 - ["Neural Machine Translation by Jointly Learning to Align and Translate"](https://arxiv.org/pdf/1409.0473v7.pdf) - Dzmitry Bahdanau, KyungHyun Cho Yoshua Bengio
* self-attention / scaled dot-product attention / transformers
  * 2017 - ["A Structured Self-attentive Sentence Embedding"](https://arxiv.org/abs/1703.03130) - Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio
  * 2017 - ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) - Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
* dropout
  * 2014 - ["Dropout: A Simple Way to Prevent Neural Networks from Overfitting"](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) - Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov
* AdaIN - see alse: StyleGAN, StyleGANv2
  * 2017 - ["Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"](https://arxiv.org/abs/1703.06868) - Xun Huang, Serge Belongie


# RL

Good list here: https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html#citations-below

* multi-armed bandit
* temporal differences
  * 1988 - ["Learning to predict by the methods of temporal differences"](https://link.springer.com/content/pdf/10.1007/BF00115009.pdf) - Richard S. Sutton
* Q learning, experience replay
  * 1989 - ["Learning from Delayed Rewards"](http://www.cs.rhul.ac.uk/~chrisw/new_thesis.pdf) - Christopher Watkins
  * 2013 - (DQN) ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) - Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller
  * 2015 - ["Human-level control through deep reinforcement learning"](https://www.datascienceassn.org/sites/default/files/Human-level%20Control%20Through%20Deep%20Reinforcement%20Learning.pdf) - Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, Stig Petersen, Charles Beattie, Amir Sadik, Ioannis Antonoglou, Helen King, Dharshan Kumaran, Daan Wierstra, Shane Legg, Demis Hassabis
* Policy gradient
* proximal policy optimization (PPO)
  * https://openai.com/research/openai-baselines-ppo
  * 2017 - ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347) - John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
* direct preference optimization (DPO)
  * 2023 - ["Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290) - Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn

# Hyperparameter tuning / Architecture Search

* random search > grid search
  * 2012 - ["Random Search for Hyper-Parameter Optimization"](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a) - James Bergstra, Yoshua Bengio
* bayesian / gaussian process (explore/exploit)
  * 2009 - ["Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design"](https://arxiv.org/abs/0912.3995) - Niranjan Srinivas, Andreas Krause, Sham M. Kakade, Matthias Seeger
* Population based training
* bandit/hyperband
  * 2016 - ["Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization"](https://arxiv.org/pdf/1603.06560.pdf) - 
* Architecture search using hypernetwork proxy
  * 2017 - ["SMASH: One-Shot Model Architecture Search through HyperNetworks"](https://arxiv.org/abs/1708.05344) - Andrew Brock, Theodore Lim, J.M. Ritchie, Nick Weston

# Implicit Representation

* Occupancy Networks
  * 2018 - ["Occupancy Networks: Learning 3D Reconstruction in Function Space"](https://arxiv.org/abs/1812.03828) - Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, Andreas Geiger
* SIREN
  * 2020 - [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) - Vincent Sitzmann, Julien N. P. Martel, Alexander W. Bergman, David B. Lindell, Gordon Wetzstein
* Neural Radiance Fields (NeRF)
  * 2020 - [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) - Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng
  * 2020 - [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739) - Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng

* NeRF + Triplanes
* TensoRT

# Specific architectures/achievements, and other misc milestones

## Computer Vision / representation learning

* Video Segmentation
  * 2018 - ["Tracking Emerges by Colorizing Videos"](https://arxiv.org/abs/1806.09594) - Carl Vondrick, Abhinav Shrivastava, Alireza Fathi, Sergio Guadarrama, Kevin Murphy

* LeNet
  * 1998 - ["GradientBased Learning Applied to Document Recognition"](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) - Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner
* alexnet - Demonstrated importance of network depth (specifically stacking convolutions), and ReLU capability over the then conventional sigmoid and tanh activations
  * 2012 - (using ImageNet leaderboard date; article published 2017) - ["ImageNet Classification with Deep Convolutional Neural Networks"](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) - Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton
* GAN, DCGAN, WGAN
* StyleGAN -> StyleGANv2 -> StyleGAN2-ADA
* U-net
* VGG
  * 2014 - ["Very Deep Convolutional Networks for Large-Scale Image Recognition"](https://arxiv.org/abs/1409.1556) - Karen Simonyan, Andrew Zisserman
* inception/DeepDream
  * 2014 - ["Understanding Deep Image Representations by Inverting Them"](https://arxiv.org/abs/1412.0035) - Aravindh Mahendran, Andrea Vedaldi
  * 2015 - ["Inceptionism: Going Deeper into Neural Networks"](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) - Alexander Mordvintsev, Christopher Olah, Mike Tyka
* style transfer, content-texture decomposition, weight covariance transfer
* cyclegan/discogan
* YOLO
* EfficientNet - Scaling laws for conv-resnets
  * 2019 - ["EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf) - Mingxing Tan, Quoc V. Le
* FPN - Feature Pyramid Networks
  * 2016 - ["Feature Pyramid Networks for Object Detection"](https://arxiv.org/pdf/1612.03144.pdf) - Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie
* Mask R-CNN
  * 2017 - ["Mask R-CNN"](https://arxiv.org/pdf/1703.06870.pdf) - Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick
* Mobilenet
  * 2017 - ["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/pdf/1704.04861.pdf) - Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam
  * 2018 - ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381) - Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen
* Generative Diffusion models
  * 2015 - ["Deep Unsupervised Learning using Nonequilibrium Thermodynamics"](https://arxiv.org/abs/1503.03585) - Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli
  * 2020 - ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) - Jonathan Ho, Ajay Jain, Pieter Abbeel
* VQVAE/VQGAN
  * 2017 - ["Neural Discrete Representation Learning"](https://arxiv.org/pdf/1711.00937.pdf) - Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu
  * 2019 - ["Generating Diverse High-Fidelity Images with VQ-VAE-2"](https://arxiv.org/pdf/1906.00446.pdf) - Ali Razavi, Aaron van den Oord, Oriol Vinyals
  * 2020 - ["Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) - Patrick Esser, Robin Rombach, Björn Ommer

* Stable Diffusion
  * 2022 - ["High-Resolution Image Synthesis with Latent Diffusion Models"](https://arxiv.org/abs/2112.10752) - Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer

* ViT
  * 2020 - ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929) - Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby
  * 2022 - ["Better plain ViT baselines for ImageNet-1k"](https://arxiv.org/abs/2205.01580) - Lucas Beyer, Xiaohua Zhai, Alexander Kolesnikov

* text-to-3d, score distillation sampling, dreamfusion
  *  2022 - ["DreamFusion: Text-to-3D using 2D Diffusion"](https://arxiv.org/abs/2209.14988) - Ben Poole, Ajay Jain, Jonathan T. Barron, Ben Mildenhall

* Gaussian Splatting
  * 2023 - ["3D Gaussian Splatting for Real-Time Radiance Field Rendering"](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf)
 
* SAM
  * 2023 - ["Segment Anything"](https://arxiv.org/abs/2304.02643) - Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, Ross Girshick

* ConvNeXt
  * 2022 - ["A ConvNet for the 2020s"](https://arxiv.org/abs/2201.03545) - Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, Saining Xie

## NLP

* BERT
  * 2018 - ["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805v2) - Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
* RNN-LM
  * 2010 - ["Recurrent neural network based language model"](https://www.isca-speech.org/archive/archive_papers/interspeech_2010/i10_1045.pdf) - Tomas Mikolov, Martin Karafiat, Luka's Burget, Jan "Honza" Cernocky, Sanjeev Khudanpur
  * 2014 - ["Generating Sequences With Recurrent Neural Networks"](https://arxiv.org/pdf/1308.0850.pdf) - Alex Graves
* word2vec
  * 2013 - ["Distributed Representations of Words and Phrases and their Compositionality"](https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf) - Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean
* GLoVe
* GLUE task
* fasttext
* wordpiece tokenization / BPE
  * 2015 - ["Neural Machine Translation of Rare Words with Subword Units"](https://arxiv.org/abs/1508.07909) - Rico Sennrich, Barry Haddow, Alexandra Birch
  * 2016 - ["Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"](https://arxiv.org/abs/1609.08144v2) - Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V. Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, Jeff Klingner, Apurva Shah, Melvin Johnson, Xiaobing Liu, Łukasz Kaiser, Stephan Gouws, Yoshikiyo Kato, Taku Kudo, Hideto Kazawa, Keith Stevens, George Kurian, Nishant Patil, Wei Wang, Cliff Young, Jason Smith, Jason Riesa, Alex Rudnick, Oriol Vinyals, Greg Corrado, Macduff Hughes, Jeffrey Dean
* Large Language Models
  * ULMfit, transfer learning, llm finetuning
    * 2018 - ["Universal Language Model Fine-tuning for Text Classification"](https://arxiv.org/abs/1801.06146) - Jeremy Howard, Sebastian Ruder
  * GPT-2 / GPT-3
    * 2020 - ["Language Models are Few-Shot Learners"](https://arxiv.org/abs/2005.14165) - Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei


## Representation Learning

* autoencoders
  * 1991 - ["Nonlinear principal component analysis using autoassociative neural networks"](https://www.researchgate.net/profile/Abir_Alobaid/post/To_learn_a_probability_density_function_by_using_neural_network_can_we_first_estimate_density_using_nonparametric_methods_then_train_the_network/attachment/59d6450279197b80779a031e/AS:451263696510979@1484601057779/download/NL+PCA+by+using+ANN.pdf) - Mark A. Kramer
  * 2006 - ["Reducing the Dimensionality of Data with Neural Networks"](https://www.cs.toronto.edu/~hinton/science.pdf) - Geoff Hinton, R. R. Salakhutdinov
* VAE (w inference amortization)
  * 2013 - ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) - Diederik P Kingma, Max Welling

* siamese network
  * 2015 - ["Siamese Neural Networks for One-Shot Image Recognition"](http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf) - Gregory Koch
* student-teacher transfer learning, catastrophic forgetting
  * see also knowledge distillation below

* InfoNCE, contrastive learning
  * 2018 - ["Representation Learning with Contrastive Predictive Coding"](https://arxiv.org/abs/1807.03748) - Aaron van den Oord, Yazhe Li, Oriol Vinyals


* DINO
  * 2021 - ["Emerging Properties in Self-Supervised Vision Transformers"](https://arxiv.org/abs/2104.14294) - Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin
  * 2023 - ["DINOv2: Learning Robust Visual Features without Supervision"](https://arxiv.org/abs/2304.07193) - Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski
* CLIP
  * 2021 - [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) - Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever
  * 2022 - ["Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive Representation Learning"](https://arxiv.org/abs/2203.02053) - Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Zou


* Johnson-Lindenstrauss lemma
  * 1984 - ["Extensions_of_Lipschitz_mappings_into_a_Hilbert_space"](https://www.researchgate.net/publication/301840852) - William B. Johnson, , Joram Lindenstrauss
  * https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
  * 2021 - ["An Introduction to Johnson-Lindenstrauss Transforms"](https://arxiv.org/abs/2103.00564) - Casper Benjamin Freksen

## Misc

* Neural ODE
* Neural PDE


* seq2seq
  * 2014 - ["Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215) - Ilya Sutskever, Oriol Vinyals, Quoc V. Le
* pix2pix
* BNN
  * 1995 - ["Bayesian Methods for Neural Networks"](https://www.microsoft.com/en-us/research/wp-content/uploads/1995/01/NCRG_95_009.pdf) - Christopher Bishop 
* The Netflix Prize
  * 2007 - ["The Netflix Prize"](https://web.archive.org/web/20070927051207/http://www.netflixprize.com/assets/NetflixPrizeKDD_to_appear.pdf) - overview of the contest and dataset
  * 2007 - ["The BellKor solution to the Netflix Prize"](https://www.netflixprize.com/assets/ProgressPrize2007_KorBell.pdf) - Robert M. Bell, Yehuda Koren, Chris Volinsky
  * 2008 - ["The BellKor 2008 Solution to the Netflix Prize"](https://www.netflixprize.com/assets/ProgressPrize2008_BellKor.pdf) - Robert M. Bell, Yehuda Koren, Chris Volinsky
  * 2008 - ["The BigChaos Solution to the Netflix Prize 2008"](https://www.netflixprize.com/assets/ProgressPrize2008_BigChaos.pdf) - Andreas Toscher, Michael Jahrer
  * 2006 - ["How To Break Anonymity of the Netflix Prize Dataset"](https://arxiv.org/abs/cs/0610105) - Arvind Narayanan, Vitaly Shmatikov
* Kaggle Galaxy Zoo
* Capsule Networks
* BiDirectional RNN
  * 1997 - ["Bidirectional Recurrent Neural Networks"](https://www.researchgate.net/publication/3316656_Bidirectional_recurrent_neural_networks) - Mike Schuster, Kuldip K. Paliwal
* WaveNet
  * 2016 - ["WaveNet: A Generative Model for Raw Audio"](https://arxiv.org/pdf/1609.03499.pdf) - Aaron van den Oord, Sander Dieleman, Heiga Zen, Karen Simonyan, Oriol Vinyals, Alex Graves, Nal Kalchbrenner, Andrew Senior, Koray Kavukcuoglu
* Normalizing flows
* AlphaFold, EvoFormer
  * 2021 - ["Highly accurate protein structure prediction with AlphaFold"](https://www.nature.com/articles/s41586-021-03819-2) - John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko, Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie, Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Demis Hassabis
* AlphaGo
* IBM Watson on Jeopardy
* Understanding the GPU compute paradigm
  * 2022 - ["Making Deep Learning Go Brr From First Principles"](https://horace.io/brrr_intro.html) - Horace He


# Learning theory / Deep learning theory / model compression / interpretability / Information Geometry

* VC Dimension
  * 1971 - ["On the uniform convergence of relative frequencies of events to their probabilities"](https://courses.engr.illinois.edu/ece544na/fa2014/vapnik71.pdf) - V. Vapnik and A. Chervonenkis
  * 1989 - ["Learnability and the Vapnik-Chervonenkis Dimension "](https://www.trhvidsten.com/docs/classics/Blumer-1989.pdf) - Blumer, A.; Ehrenfeucht, A.; Haussler, D.; Warmuth, M. K. 
* gradient double descent
  * 2019 - ["Deep Double Descent: Where Bigger Models and More Data Hurt"](https://arxiv.org/abs/1912.02292) - Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, Ilya Sutskever
  * 2019 - ["Reconciling modern machine-learning practice and the classical bias–variance trade-off"](https://www.pnas.org/doi/10.1073/pnas.1903070116) - Mikhail Belkin, Daniel Hsu, Siyuan Ma, and Soumik Mandal
  * 2020 - ["The generalization error of random features regression: Precise asymptotics and double descent curve](https://arxiv.org/abs/1908.05355) - Song Mei, Andrea Montanari
  * 2021 -["A Farewell to the Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized Machine Learning"](https://arxiv.org/pdf/2109.02355.pdf) - Yehuda Dar, Vidya Muthukumar, Richard G. Baraniuk
  * 1999 - ["Generalization in a linear perceptron in the presence of noise"](https://www.researchgate.net/publication/231084158_Generalization_in_a_linear_perceptron_in_the_presence_of_noise) - Anders Krogh, John A Hertz
  * 2020 - ["The Neural Tangent Kernel in High Dimensions: Triple Descent and a Multi-Scale Theory of Generalization"](http://proceedings.mlr.press/v119/adlam20a/adlam20a.pdf) - Ben Adlam, Jeffrey Pennington

* neural tangent kernel
  * 2018 - ["Neural Tangent Kernel: Convergence and Generalization in Neural Networks"](https://arxiv.org/abs/1806.07572) - Arthur Jacot, Franck Gabriel, Clément Hongler

* lottery ticket hypothesis
  * 2018 - ["The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"](https://arxiv.org/abs/1803.03635) - Jonathan Frankle, Michael Carbin

* manifold hypothesis
* information bottleneck
  * 2015 - ["Deep Learning and the Information Bottleneck Principle"](https://arxiv.org/abs/1503.02406) - Naftali Tishby, Noga Zaslavsky
* generalized degrees of freedom
  * 1998 - "On Measuring and Correcting the Effects of Data Mining and Model Selection" - Jianming Ye
* AIC / BIC
* dropout as ensemblification
  * 2013 - ["Understanding Dropout"](https://papers.nips.cc/paper/2013/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf) - Pierre Baldi, Peter Sadowski
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
* RNN's are near-sighted
  * 2003 - ["Gradient Flow in Recurrent Nets: the Difficulty of Learning Long-Term Dependencies"](http://www.bioinf.jku.at/publications/older/ch7.pdf) - Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, Jurgen Schmidhuber
  * 2014 - ["On the Properties of Neural Machine Translation: Encoder–Decoder Approaches"](https://arxiv.org/pdf/1409.1259.pdf) - 
* Relationship between logistic regression and naive bayes
  * 2002 - ["On Discriminative vs. Generative classifiers: A comparison of logistic regression and naive Bayes](https://proceedings.neurips.cc/paper/2001/file/7b7a53e239400a13bd6be6c91c4f6c4e-Paper.pdf) - Andrew Ng, Michael Jordan
* understanding softmax and its relation to log-sum-exp
  * 2017 - ["On the Properties of the Softmax Function with Application in Game Theory and Reinforcement Learning"](https://arxiv.org/pdf/1704.00805.pdf) - Bolin Gao, Lacra Pavel
* Word2vec approximates a matrix factorization
  * 2014 - ["Neural Word Embedding as Implicit Matrix Factorization"](https://papers.nips.cc/paper/2014/file/feab05aa91085b7a8012516bc3533958-Paper.pdf) - Omer Levy, Yoav Goldberg
* The distributional hypothesis (computational linguistics)
  * 1954 - ["Distributional Structure"](https://www.tandfonline.com/doi/pdf/10.1080/00437956.1954.11659520) - Zellig Harris
* Johnson–Lindenstrauss lemma (high dim manifolds can be accurately projected onto lower dim embeddings)
  * 1984 - "Extensions of Lipschitz mappings into a Hilbert space" - William B. Johnson, Joram Lindenstrauss
  * https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma
* Empirical Risk Minimization
  * 1992 - ["Principles of Risk Minimization for Learning Theory"](https://proceedings.neurips.cc/paper/1991/file/ff4d5fbbafdf976cfdc032e3bde78de5-Paper.pdf) - V. Vapnik

* Loss geometry
  * 2014 - ["Identifying and attacking the saddle point problem in high-dimensional non-convex optimization"](https://arxiv.org/abs/1406.2572) - Yann Dauphin, Razvan Pascanu, Caglar Gulcehre, Kyunghyun Cho, Surya Ganguli, Yoshua Bengio

* generalization, overparameterization, effective model capacity, grokking
  * 2016 - ["Understanding deep learning requires rethinking generalization"](https://arxiv.org/pdf/1611.03530.pdf) - Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, Oriol Vinyals
  * 2021 - [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) - Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, Vedant Misra
  * 2022 - [Towards Understanding Grokking: An Effective Theory of Representation Learning](https://arxiv.org/abs/2205.10343) - Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, Mike Williams


* Strong inductive bias in CNN structure 
  * 2018 - ["Deep Image Prior"](https://openaccess.thecvf.com/content_cvpr_2018/html/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.html) - Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky

* prediction calibration
  * 2017 - ["On Calibration of Modern Neural Networks"](http://proceedings.mlr.press/v70/guo17a.html) - Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger

* batch-norm and dropout in tug-of-war
  * 2018 - ["Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift"](https://arxiv.org/pdf/1801.05134.pdf) - Xiang Li, Shuo Chen, Xiaolin Hu, Jian Yang

* deep learning model fitting process
  * http://karpathy.github.io/2019/04/25/recipe/  

* Universal approximation theorem
  * 1989 - ["Multilayer Feedforward Networks are Universal Approximators"](https://cognitivemedium.com/magic_paper/assets/Hornik.pdf) -  Hornik, Kurt; Tinchcombe, Maxwell; White, Halbert
  * 1991 - ["Approximation capabilities of multilayer feedforward networks"](https://web.njit.edu/~usman/courses/cs677_spring21/hornik-nn-1991.pdf) -  Kurt Hornik
  * 1993 - ["Multilayer Feedforward Networks With a Nonpolynomial Activation Function Can Approximate Any Function"](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.145.6041&rep=rep1&type=pdf) - Moshe Leshno, Lin Vladimir Ya, Allan Pinkus, Shimon Schocken

* Dropout as approximate bayesian inference
  * 2015 - ["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"](https://arxiv.org/pdf/1506.02142.pdf) - Yarin Gal, Zoubin Ghahramani

* CMA-ES learns an approximation to the hessian of the loss
  * 2019 - ["On the covariance-Hessian relation in evolution strategies"](https://www.sciencedirect.com/science/article/abs/pii/S0304397519305468) - Ofer M. Shir, Amir Yehudayoff

* Inductive Biases
  * 1980 - ["The need for biases in learning generalizations"](http://www.cs.cmu.edu/~tom/pubs/NeedForBias_1980.pdf) - Tom Mitchell
  * 2018 - ["Relational inductive biases, deep learning, and graph networks"](https://arxiv.org/pdf/1806.01261.pdf) - Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu

* Neural Networks are essentially high dimensional decision trees. The latent space can be datum-specific. The learned manifold is not smooth, but heavily faceted. each neuron (relu) adds a hyperplane.
  * 2018 - [A Spline Theory of Deep Networks](https://proceedings.mlr.press/v80/balestriero18b/balestriero18b.pdf) - Randall Balestriero Richard G. Baraniuk
* Extrapolation
  * 2021 - ["Learning in High Dimension Always Amounts to Extrapolation"](https://arxiv.org/pdf/2110.09485.pdf) - Randall Balestriero, J´erˆome Pesenti, and Yann LeCun
  
* Formalizing "intelligence"
  * 2019 - ["On the Measure of Intelligence"](https://arxiv.org/abs/1911.01547) - François Chollet

* PEFT: LoRA
  * 2021 - ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) - Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
* Fourier features

* NoPe - positional encodings not needed, learned implicitly
  * 2023 - ["The Impact of Positional Encoding on Length Generalization in Transformers"](https://arxiv.org/abs/2305.19466) - Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, Siva Reddy

* Intrinsic Dimension, PEFT
  * 2020 - ["Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning"](https://arxiv.org/abs/2012.13255) - Armen Aghajanyan, Luke Zettlemoyer, Sonal Gupta

* GAN training dynamics, TTUR
  * 2017 - ["GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"](https://arxiv.org/abs/1706.08500v6) - Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter

* The Bitter Lesson
  * 2019 - ["The Bitter Lesson"](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - Rich Sutton

* sigma reparameterization to stabilize transformer training by mitigating "entropy collapse" (concentration of density in attention)
  * 2023 - ["Stabilizing Transformer Training by Preventing Attention Entropy Collapse"](https://arxiv.org/abs/2303.06296) - Shuangfei Zhai, Tatiana Likhomanenko, Etai Littwin, Dan Busbridge, Jason Ramapuram, Yizhe Zhang, Jiatao Gu, Josh Susskind

* buffer tokens, register tokens, attention sinks
  * 2023 - ["Vision Transformers Need Registers"](https://arxiv.org/abs/2309.16588) - Timothée Darcet, Maxime Oquab, Julien Mairal, Piotr Bojanowski
  * 2023 - ["Efficient Streaming Language Models with Attention Sinks"](https://arxiv.org/abs/2309.17453) - Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis

## Surprisingly Relevant Group Theory

* Connecting the Variational Renormalization Group and Unsupervised Learning
  * 2014 - ["An exact mapping between the Variational Renormalization Group and Deep Learning"](https://arxiv.org/abs/1410.3831) - Pankaj Mehta, David J. Schwab
* learning a feature is equivalent to searching for a transformation that stabilizes it.
  * 2015 - ["Why does Deep Learning work? - A perspective from Group Theory"](https://arxiv.org/abs/1412.6621) - Arnab Paul, Suresh Venkatasubramanian

# Information theory

* Entropy
  * 1948 - ["A Mathematical Theory of Communication"](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) - Claude Shannon
* Fisher information
* KL divergence


# Causal Modeling / experimentation

* Double machine learning
  * 2016 - ["Double/Debiased Machine Learning for Treatment and Causal Parameters"](https://arxiv.org/abs/1608.00060) - Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins
* Doubly robust inference
* Pearl's do calculus and graphical modeling  / structural equation modeling
* Rubin's potential outcomes model
* model identification
* d-separation
* propensity scoring/matching
* item-response model and adaptive testing
* bandit learning for on-line experimentation
* belief propagation

# Time series forecasting

* ARMA / ARIMA / ARIMAX
* sin/cos cyclical day encodings
* RNN forecasting
  * 1991 - ["Recurrent Networks and NARMA Modeling"](https://proceedings.neurips.cc/paper/1991/file/5ef0b4eba35ab2d6180b0bca7e46b6f9-Paper.pdf) - J. Connor, L. Atlas, R. Martin
  * 2017 - ["A Multi-Horizon Quantile Recurrent Forecaster"](https://arxiv.org/pdf/1711.11053.pdf) - (Amazon) Ruofeng Wen, Kari Torkkola, Balakrishnan Narayanaswamy, Dhruv Madeka
* FB Prophet / bayesian

# Misc Generative Art milestones and techniques

* Utilizing optical flow to stabilize synthesized video frames
  * 2016 - ["Artistic style transfer for videos"](https://arxiv.org/pdf/1604.08610.pdf) - Manuel Ruder, Alexey Dosovitskiy, Thomas Brox

# Ethics in ML

* Data Privacy
  * See Netflix Prize 
* Differential Privacy
* k-anonymity
* Dataset bias - gendered words, differential treatment of skin color, race and zipcode in legal applications
* YOLO author's resignation (blog post + reddit thread)
* CV techniques used to subjugate minorities in SE Asia and China
* Ethical issues surrounding classification of behavioral health and interventions
* Metadata deanonymization and leaks of US domestic data collection programs with corporate participation
* "fairness" algorithms
* gerrymandering and algorithmic redistricting
* Facebook's influence on elections and live-testing to influence people's emotions and behaviors w/o consent

# Analytic Process

* ML Tech Debt
  * 2015 - ["Hidden Technical Debt in Machine Learning Systems"](https://proceedings.neurips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) - D. Sculley, Gary Holt, Daniel Golovin, Eugene Davydov, Todd Phillips, Dietmar Ebner, Vinay Chaudhary, Michael Young, Jean-Franc¸ois Crespo, Dan Dennison

# Misc important papers for generative models/art, misc modern era

* Classifier-free Guidance (CFG)
  * 2021 - [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) - Jonathan Ho, Tim Salimans

* SDEdit
  * 2021 - ["SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations"](https://arxiv.org/abs/2108.01073) - Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, Stefano Ermon

* Denoising diffusion as generic de-corruption
  * 2022 - ["Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise"](https://arxiv.org/abs/2208.09392) - Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie S. Li, Hamid Kazemi, Furong Huang, Micah Goldblum, Jonas Geiping, Tom Goldstein
* k-samplers, variance-preserving, variance exploding
  * 2022 - ["Elucidating the Design Space of Diffusion-Based Generative Models"](https://arxiv.org/abs/2206.00364) - Tero Karras, Miika Aittala, Timo Aila, Samuli Laine

* Cross Attention guidance
* Controlnet/T2I adaptors
* Text inversion
* null text inversion


# LLMs, in-context learning, prompt engineering

* Chain of thought
  * 2022 - ["Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"](https://arxiv.org/abs/2201.11903) - Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou

* LLMs as role-play
  * 2023 - ["Role-Play with Large Language Models"](https://arxiv.org/abs/2305.16367) - Murray Shanahan, Kyle McDonell, Laria Reynolds
  
* learning to use tools
  * 2023 - ["Toolformer: Language Models Can Teach Themselves to Use Tools"](https://arxiv.org/abs/2302.04761) - Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom
    
* Chinchilla
  * 2022 - ["Training Compute-Optimal Large Language Models"](https://arxiv.org/abs/2203.15556) - Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre

* Instruct tuning, InstructGPT
  * 2022 - ["Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155) - Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe

* Speculative decoding
  * 2023 -["Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) - Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper

# Development of attention mechanisms

largely via https://twitter.com/karpathy/status/1668302116576976906

* https://arxiv.org/abs/1308.0850
* https://arxiv.org/abs/1409.0473
* https://arxiv.org/abs/1410.5401
* Attention is all you need
