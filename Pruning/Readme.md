# Network Sparsification (Pruning)

[TOC]

> Thanks for 
>
> - [github: he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
>
> - [模型加速与压缩 | 剪枝乱炖](https://zhuanlan.zhihu.com/p/97198052)

![](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20201014115617.png)



# 1 基于度量标准的剪枝

这类方法通常是提出一个判断神经元是否重要的度量标准，依据这个标准计算出衡量神经元重要性的值，将不重要的神经元剪掉。在神经网络中可以用于度量的值主要分为：**Weight / Activation / Gradient / Error**

## 1.1 Channels

> include Weight and Gradient

### Filter pruning

- [Channel Pruning via Automatic Structure Search](https://arxiv.org/abs/2001.08565) | [PyTorch(Author)](https://github.com/lmbxmu/ABCPruner) [arXiv  '20]

- [HRank: Filter Pruning using High-Rank Feature Map](http://arxiv.org/abs/2002.10179) | [code](https://github.com/lmbxmu/HRank)  [arXiv  '20]

- [Filter sketch for network pruning](http://arxiv.org/abs/2001.08514)  | [code](https://github.com/lmbxmu/FilterSketch)  [arXiv  '20]

- [Model compression using progressive channel pruning](https://sci-hub.se/downloads/2020-06-02/57/10.1109@TCSVT.2020.2996231.pdf?rand=5f745440eee93?download=true)  [IEEE  '20]

- [Play and Prune: Adaptive Filter Pruning for Deep Model Compression](https://arxiv.org/abs/1905.04446)   [CVPR '19]

- [Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks](https://arxiv.org/abs/1909.08174)    [CVPR '19]

- **Filter pruning via geometric median for deep convolutional neural networks acceleration   [CVPR '19]**

- [Compressing convolutional neural networks via factorized convolutional filters](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.pdf)   [CVPR '19]

- Centripetal SGD for pruning very deep convolutional networks with complicated structure.  [arXiv  '19]

- Discrimination-aware channel pruning for deep neural networks  [NIPS '18]

- [2PFPCE: Two-phase filter pruning based on conditional entropy ](http://arxiv.org/abs/1809.02220) [arXiv '18]

- [Layer-compensated pruning for resource-constrained convolutional neural networks](http://arxiv.org/abs/1810.00518)   [arXiv '18]

- [RePr: Improved Training of Convolutional Filters](https://arxiv.org/pdf/1811.07275.pdf)  [arXiv '18]

- **[Structured Pruning of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1512.08571)  [arXiv  '15]**

  > feature map; using the evolutionary particle fifiltering approach

### Weight pruning

- [Importance Estimation for Neural Network Pruning](https://arxiv.org/abs/1906.10771)   [CVPR '19]

- [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)  [ICLR '19]

- [NeST: A neural network synthesis tool based on a grow-and-prune paradigm](https://arxiv.org/abs/1711.02017)  [CVPR '19]

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)   [ICLR '19]

  > The lottery ticket hypothesis [5] sets the weights below a threshold to zero, rewinds the rest of the weights to their initial confifiguration, and then retrains the network from this confifiguration.

- Learning-compression algorithms for neural net pruning   [CVPR '18]

- [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)   [ICLR '18]

- [A novel channel pruning method for deep neural network compression](https://arxiv.org/abs/1805.11394)  [CVPR '18]

- [Faster gaze prediction with dense networks and Fisher pruning](https://arxiv.org/abs/1801.05787)    [CVPR '18]

- Netadapt: Platform-aware neural network  adaptation  for  mobile  applications [ECCV '18]

- [StructADMM: A systematic, high-efficiency framework of structured weight pruning for DNNs](http://arxiv.org/abs/1807.11091)   [arXiv '18]

- [Progressive weight pruning of deep neural networks using ADMM](https://arxiv.org/abs/1810.07378)  [CVPR  '18]

- [Soft Weight-Sharing for Neural Network Compression](https://arxiv.org/abs/1702.04008) [ICLR '17]

- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)  [ICLR '17]  | l1-norm

- [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)  [ICLR '17]  | Taylor

- [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128)  [CVPR '17]

- Net-trim:  Convex pruning of deep neural networks with performance guarantee  [NIPS '17]

- [Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)  [arXiv  '16]

- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)  [ICLR '16]  | Threshold | first

## 1.2 Neuron

> include Activation and Error

- Frequency-domain dynamic pruning for convolutional neural networks [NIPS '18]

- [An entropy-based pruning method for cnn compression](https://arxiv.org/abs/1706.05791).   [arXiv '17]

- **[Scalpel: Customizing DNN Pruning to the Underlying Hardware Parallelism](http://www-personal.umich.edu/~jiecaoyu/papers/jiecaoyu-isca17.pdf) [17]**

  > proposed a neuron pruning method by introducing a binary mask sampled from a trainable scaling factor for each FM

- **[Dynamic Network Surgery for Efficient DNNs](https://arxiv.org/abs/1608.04493) [NIPS '16]**

- **[Learning both Weights and Connections for Efficient Neural Network](https://arxiv.org/abs/1506.02626) | 引用 2546   [arXiv  '15] | l2-norm | first**

- [Network trimming: A data-driven neuron pruning approach towards efficient deep architectures](https://arxiv.org/abs/1607.03250)   [arXiv '16] | rank

# 2 基于重建误差的剪枝

> pre-training --> keep the most important filters weights --> fine-tuning
>
> 这类方法通过最小化特征输出的重建误差来确定哪些filters要进行剪裁，即找到当前层对后面的网络层输出没啥影响的信息。

- [Soft filter pruning for accelerating deep convolutional neural networks](http://arxiv.org/abs/1808.06866)  [arXiv '18] | l1-norm
- Discrimination-aware channel pruning for deep neural networks  [NIPS  '2017]
- **[Channel pruning for accelerating very deep neural networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf) [ICCV '17] | lasso**
- **[ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) [ICCV '17] | greedy algorithm**
- [Pruning  filters  for efficient convnet](https://arxiv.org/abs/1608.08710)  [ICLR '17] | sparsity of outputs
- [NISP: Pruning Networks Using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908)  [arXiv  '17]
- Learning to prune deep neural networks via layerwise optimal brain surgeon [NIPS  '2017] | Taylor | layer-wise surgeon
- [Deep residual learning for image recognition](https://arxiv.org/abs/1512.03385)  [CVPR '16]

# 3 基于稀疏训练的剪枝

> 这类方法采用训练的方式，结合各种regularizer来让网络的权重变得稀疏，于是可以将接近于0的值剪掉。

- [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935) | [code](https://github.com/ofsoundof/group_sparsity)  [CVPR '20]
- [Towards optimal structured cnn pruning via generative adversarial learning](https://arxiv.org/abs/1903.09291)   [CVPR '19]
- Variational convolutional neural network pruning   [CVPR '19]
- [Rethinking the smaller-norm-less-informative assumption in channel pruning of convolution layers](http://arxiv.org/abs/1802.00124)   [ICLR '19] | l1
- [Full deep neural network training on a pruned weight budget](https://arxiv.org/pdf/1806.06949)  [cs.LG  '19]
- [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/pdf/1707.01213.pdf)   [ECCV '18]
- [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.06798)    [CVPR '18]
- [Compression of deep convolutional neural networks under joint sparsity constraints](http://arxiv.org/abs/1805.08303)   [CVPR '18]
- [NestedNet: Learning nested sparse structures in deep neural networks](https://arxiv.org/abs/1712.03781)   [CVPR '18]
- [A systematic DNN weight pruning framework using alternating direction method of multipliers](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.pdf)   [ECCV '18]
- [Tetris: Tile-matching the tremendous irregular sparsity](https://papers.nips.cc/paper/7666-tetris-tile-matching-the-tremendous-irregular-sparsity)  [NIPS '18]  | block-wise weight sparsity
- [Hybrid pruning: Thinner sparse networks for fast inference on edge devices](http://arxiv.org/abs/1811.00482)    [arXiv '18]
- **[Deep gradient compression: Reducing the communication bandwidth for distributed training](https://arxiv.org/abs/1712.01887)  [arXiv  '17]**
- **[Learning efficient convolutional networks through network slimming](https://arxiv.org/abs/1708.06519)   [ICCV '17] | L1 regularization on BN | time **
- [Exploring sparsity in recurrent neural networks](http://arxiv.org/abs/1704.05119)  [arXiv  '17]
- [Less Is More: Towards Compact CNNs](http://users.umiacs.umd.edu/~hzhou/paper/zhou_ECCV2016.pdf)  [ECCV  '16]
- **[Learning Structured Sparsity in Deep Neural Networks](https://arxiv.org/pdf/1608.03665.pdf)   [arXiv  '16]**  | layer

# 4 Random and Rethinking

> 有采用各种剪枝方法的就有和这些剪枝方法对着干的。

- [Rethinking the Value of Network Pruning](https://zhuanlan.zhihu.com/write)  [ICLR '19]

  >  the structure of the pruned model is more important than the inherited “important” weights

- [Recovering from Random Pruning: On the Plasticity of Deep Convolutional Neural Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1801.10447.pdf) [arXiv  '18]

- [Pruning from Scratch](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.12579)  [arXiv  '19]

# 5 走向NAS的自动化剪枝

- [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://arxiv.org/abs/2005.03354) | [code](https://github.com/zx55/dmcp) [CVPR  '20] | NAS + Channel Pruning

- [Metapruning:  Meta learning for automatic neural network channel pruning](https://arxiv.org/abs/1903.10258)  [ICCV '19] | first automatic

- Network pruning via transformable architecture search.   2019.

- [Autoprune: Automatic network pruning by regularizing auxiliary parameters](http://papers.nips.cc/paper/9521-autoprune-automatic-network-pruning-by-regularizing-auxiliary-parameters.pdf)   [NIPS  '19]

- [AutoPruner: An end-to-end trainable filter pruning method for efficient deep model inference](http://arxiv.org/abs/1805.08941)   [arXiv '18]

- Constraint-aware deep neural network compression  [ECCV  '18]

- [AMC: Automl for model compression and acceleration on mobile devices](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf) [ECCV  '18]

  > 将强化学习引入剪枝

- Auto-balanced filter pruning for efficient convolutional neural networks  [AAAI '18]

- Runtime neural pruning  [NIPS  '17]
