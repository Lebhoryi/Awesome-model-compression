# Network Sparsification (Pruning)

[TOC]

![](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20201026155007.png)

> Thanks for 
>
> - [github: he-y/Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
> - [模型加速与压缩 | 剪枝乱炖](https://zhuanlan.zhihu.com/p/97198052)
> - [闲话模型压缩之网络剪枝（Network Pruning）篇](https://blog.csdn.net/jinzhuojun/article/details/100621397)
> - [2019-04-30-神经网络压缩综述](https://github.com/glqglq/glqblog/blob/master/_posts/2019-04-30-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%8E%8B%E7%BC%A9%E7%BB%BC%E8%BF%B0.md)

![](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20201014115617.png)

> ** **To be careful** **: 
>
> - All the papers with `|` are more important...
>
> - There are something wrong in the follow picture, I would fix it in the future. Welcome to submit pr...
>
>   <img src="https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20201026154606.png" style="zoom:80%;" />

# 0 鼻祖论文

- Comparing Biases for Minimal Network Construction with Back-Propagation | magnitude-based
- Optimal brain damage  | OBD
- Second order derivatives for network pruning: Optimal Brain Surgeon  | OBS

# 1 基于度量标准的剪枝

这类方法通常是提出一个判断神经元是否重要的度量标准，依据这个标准计算出衡量神经元重要性的值，将不重要的神经元剪掉。在神经网络中可以用于度量的值主要分为：**Weight / Activation / Gradient / Error**

## 1.1 Channels

> include Weight and Gradient

### Filter pruning

- [Towards Optimal Filter Pruning with Balanced Performance and Pruning Speed](https://arxiv.org/abs/2010.06821)  [CVPR  '20]

- [Channel Pruning via Automatic Structure Search](https://arxiv.org/abs/2001.08565) | [PyTorch(Author)](https://github.com/lmbxmu/ABCPruner) [CVPR  '20]

- [Learning Filter Pruning Criteria for Deep Convolutional Neural Networks Acceleration](https://openaccess.thecvf.com/content_CVPR_2020/html/He_Learning_Filter_Pruning_Criteria_for_Deep_Convolutional_Neural_Networks_Acceleration_CVPR_2020_paper.html)     [CVPR '20]

- [HRank: Filter Pruning using High-Rank Feature Map](http://arxiv.org/abs/2002.10179) | [code](https://github.com/lmbxmu/HRank)  [arXiv  '20]

- [Filter sketch for network pruning](http://arxiv.org/abs/2001.08514)  | [code](https://github.com/lmbxmu/FilterSketch)  [arXiv  '20]

- [Model compression using progressive channel pruning](https://sci-hub.se/downloads/2020-06-02/57/10.1109@TCSVT.2020.2996231.pdf?rand=5f745440eee93?download=true)  [IEEE  '20]

- Progressive local filter pruning for image retrieval acceleration  [arXiv  '20]

- [Importance Estimation for Neural Network Pruning](https://arxiv.org/abs/1906.10771)   [CVPR '19]

- [Play and Prune: Adaptive Filter Pruning for Deep Model Compression](https://arxiv.org/abs/1905.04446)   [CVPR '19] | 

- [Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks](https://arxiv.org/abs/1909.08174)    [CVPR '19] | GBN | FLOPs

- Filter pruning via geometric median for deep convolutional neural networks acceleration   [CVPR '19]  | magnitude-based | FPGM

- Filter- based deepcompression with global average pooling for convolutional networks 2019

- [Compressing convolutional neural networks via factorized convolutional filters](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Compressing_Convolutional_Neural_Networks_via_Factorized_Convolutional_Filters_CVPR_2019_paper.pdf)   [CVPR '19]

- Centripetal SGD for pruning very deep convolutional networks with complicated structure.  [arXiv  '19]

- Pruning filter via geometric median for deep convolutional neural networks acceleration  [CVPR '19]

- Meta filter pruning to accelerate deep convolutional neural networks    [arXiv  '19]

- Asymptotic soft filter pruning for deep convolutional neural networks  [IEEE '2019]

- [2PFPCE: Two-phase filter pruning based on conditional entropy ](http://arxiv.org/abs/1809.02220) [arXiv '18]

- [Layer-compensated pruning for resource-constrained convolutional neural networks](http://arxiv.org/abs/1810.00518)   [arXiv '18]

- [RePr: Improved Training of Convolutional Filters](https://arxiv.org/pdf/1811.07275.pdf)  [arXiv '18]

- [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)  [ICLR '17]  | Taylor

- [Pruning Filters for Efficient ConvNets](https://arxiv.org/abs/1608.08710)  [ICLR '17]  | 权重的绝对值 | **filter pruning 开山之作**

  > 不会导致不规则稀疏连接。因此，不需要稀疏卷积库的支持。为了在多个层同时进行剪枝，又提出独立剪枝和贪婪剪枝两种方式

- [Structured Pruning of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1512.08571)  [arXiv  '15] | **结构化剪枝开山之作**

  > feature map; using the evolutionary particle fifiltering approach
  >
  >  方法的主要思想是定 义显著性变量并进行贪婪剪枝，提出核内定步长粒 度将细粒度剪枝转化为粗粒度剪枝如通道剪枝或卷 积核剪枝．

### Weight pruning

- [Structured Compression by Weight Encryption for Unstructured Pruning and Quantization](https://arxiv.org/abs/1905.10138)   [CVPR '20]

- [EagleEye: Fast Sub-net Evaluation for Efficient Neural Network Pruning](https://arxiv.org/abs/2007.02491)  | [code](https://github.com/anonymous47823493/EagleEye)    [ECCV '20]

- [Differentiable Joint Pruning and Quantization for Hardware Efficiency](https://arxiv.org/abs/2007.10463)    [ECCV '20]  first

- [Multi-Dimensional Pruning: A Unified Framework for Model Compression](https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_Multi-Dimensional_Pruning_A_Unified_Framework_for_Model_Compression_CVPR_2020_paper.html)    [CVPR '20] first

- [SNIP: Single-shot Network Pruning based on Connection Sensitivity](https://arxiv.org/abs/1810.02340)  [ICLR '19]

- [NeST: A neural network synthesis tool based on a grow-and-prune paradigm](https://arxiv.org/abs/1711.02017)  [CVPR '19]

- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)   [ICLR '19] |

  > The lottery ticket hypothesis [5] sets the weights below a threshold to zero, rewinds the rest of the weights to their initial confifiguration, and then retrains the network from this confifiguration.
  >
  > 提出The Lottery Ticket Hypothesis，即一个随机初始化，密集的网络包含一个子网络，这个子网络如果沿用原网络的权重初始化，在至多同样迭代次数训练后就可以比肩原网络的测试精度。同时它还给出了找这种子网络结构的方法。文章认为这个子结构和它的初始值对训练的有效性至关重要，它们被称为『winning logttery tickets』。

- Learning-compression algorithms for neural net pruning   [CVPR '18] | 

  > 提出Learning和Compression两步交替优化的pruning方法，在Compression操作中，通过将原参数向约束表示的可行集投影来自动找到每层的最优sparsity ratio。因为此类方法不需要计算量较大的sensitivity analysis，也减少了超参数的引入。

- [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878)   [ICLR '18] |

- [A novel channel pruning method for deep neural network compression](https://arxiv.org/abs/1805.11394)  [CVPR '18]  | 遗传算法

- [Faster gaze prediction with dense networks and Fisher pruning](https://arxiv.org/abs/1801.05787)    [CVPR '18]  二阶taylor

- Netadapt: Platform-aware neural network  adaptation  for  mobile  applications [ECCV '18]  | Greedy strategy

- [StructADMM: A systematic, high-efficiency framework of structured weight pruning for DNNs](http://arxiv.org/abs/1807.11091)   [arXiv '18]

- [Progressive weight pruning of deep neural networks using ADMM](https://arxiv.org/abs/1810.07378)  [CVPR  '18]

- Deep network compression learning by in-parallel pruning-quantizatio   [CVPR  '18]

- [A systematic DNN weight pruning framework using alternating direction method of multipliers](https://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyun_Zhang_A_Systematic_DNN_ECCV_2018_paper.pdf)   [ECCV '18]

- Neural network pruning based on weight similarity 2018 | 权值相似性

  > 实现全连接层的压缩，参数减少76.83%，精度降低0.1%，预算量没有降低，证明运算量集中在卷积层

- [Soft Weight-Sharing for Neural Network Compression](https://arxiv.org/abs/1702.04008) [ICLR '17]

- Learning to prune deep neural networks via layer-wise optimal brain surgeon  [NIPS '2017]

- Fine-pruning: Joint fifine-tuning and compression of a convolutional network with bayesian ptimization | 贝叶斯

- [Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning](https://arxiv.org/abs/1611.05128)  [CVPR '17] | 能耗 + 最小二乘法

- Net-trim:  Convex pruning of deep neural networks with performance guarantee  [NIPS '17]

- [Dynamic Network Surgery for Efficient DNNs](https://arxiv.org/abs/1608.04493) [NIPS '16] | 

- [Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)  [arXiv  '16]

- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)  [ICLR '16]  | Threshold | first

- [Learning both Weights and Connections for Efficient Neural Network](https://arxiv.org/abs/1506.02626) | 引用 2546   [arXiv  '15] | l2-norm | 经典

## 1.2 Neuron

> include Activation and Error

- [Accelerating CNN Training by Pruning Activation Gradients](https://arxiv.org/abs/1908.00173)   [ECCV '20]

- Frequency-domain dynamic pruning for convolutional neural networks [NIPS '18]

- [An entropy-based pruning method for cnn compression](https://arxiv.org/abs/1706.05791).   [arXiv '17]

- **[Scalpel: Customizing DNN Pruning to the Underlying Hardware Parallelism](http://www-personal.umich.edu/~jiecaoyu/papers/jiecaoyu-isca17.pdf) [17]**

  > proposed a neuron pruning method by introducing a binary mask sampled from a trainable scaling factor for each FM

- [Network trimming: A data-driven neuron pruning approach towards efficient deep architectures](https://arxiv.org/abs/1607.03250)   [arXiv '16] | Average Percentage of Zeros | APoZ


# 2 基于重建误差的剪枝

> pre-training --> keep the most important filters weights --> fine-tuning
>
> 这类方法通过最小化特征输出的重建误差来确定哪些filters要进行剪裁
>
> 如果对当前层进行裁剪，然后如果它对后面输出还没啥影响，那说明裁掉的是不太重要的信息

- Discrimination-aware channel pruning for deep neural networks  [NIPS '18]  | 判别力驱动损失函数 + 贪婪算法

- [Soft filter pruning for accelerating deep convolutional neural networks](http://arxiv.org/abs/1808.06866)  [arXiv '18] | l1-norm | SFP

  > 此方法允许被剪掉滤波器在训练中继续更新，这使得网络有更大模型容量与优化空间，并且可以减少对预训练模型依赖，使得模型能从头训练

- Discrimination-aware channel pruning for deep neural networks  [NIPS  '2017]  | DCP

- [NISP: Pruning Networks Using Neuron Importance Score Propagation](https://arxiv.org/abs/1711.05908)  [arXiv  '17] |

  > 通过最小化分类网络倒数第二层的重建误差，并将重要性信息反向传播到前面以决定哪些channel需要裁剪

- [Channel pruning for accelerating very deep neural networks](http://openaccess.thecvf.com/content_ICCV_2017/papers/He_Channel_Pruning_for_ICCV_2017_paper.pdf) [ICCV '17] | lasso + 最小二乘法

- [ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression](https://arxiv.org/abs/1707.06342) [ICCV '17] | greedy algorithm | 经典

  > 提出将滤波器剪枝当作一个优化问题，并且揭示对滤波器进行剪枝需要基于从下一层计算得到统计信息而非当前层，这是此方法与其他方法的重要区别

- [Pruning  filters  for efficient convnet](https://arxiv.org/abs/1608.08710)  [ICLR '17] | sparsity of outputs

- Learning to prune deep neural networks via layerwise optimal brain surgeon [NIPS  '2017] | Taylor | layer-wise surgeon


# 3 基于稀疏训练的剪枝

> 这类方法采用训练的方式，结合各种regularizer来让网络的权重变得稀疏，于是可以将接近于0的值剪掉。

- [DSA: More Efficient Budgeted Pruning via Differentiable Sparsity Allocation](https://arxiv.org/abs/2004.02164)   [ECCV '20]

- [Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression](https://arxiv.org/abs/2003.08935) | [code](https://github.com/ofsoundof/group_sparsity)  [CVPR '20]

- [Towards optimal structured cnn pruning via generative adversarial learning](https://arxiv.org/abs/1903.09291)   [CVPR '19] | 生成对抗学习

- Toward compact ConvNets via structure-sparsity regularized filter pruning  2019 

- Variational convolutional neural network pruning   [CVPR '19] |  scaling factor of BN

- [Rethinking the smaller-norm-less-informative assumption in channel pruning of convolution layers](http://arxiv.org/abs/1802.00124)   [ICLR '19] | l1

- [Full deep neural network training on a pruned weight budget](https://arxiv.org/pdf/1806.06949)  [cs.LG  '19] |  magnitude of gradients

- [Data-Driven Sparse Structure Selection for Deep Neural Networks](https://arxiv.org/pdf/1707.01213.pdf)   [ECCV '18]

- [MorphNet: Fast & Simple Resource-Constrained Structure Learning of Deep Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.06798)    [CVPR '18]

- [Compression of deep convolutional neural networks under joint sparsity constraints](http://arxiv.org/abs/1805.08303)   [CVPR '18]

- [NestedNet: Learning nested sparse structures in deep neural networks](https://arxiv.org/abs/1712.03781)   [CVPR '18]

- [Tetris: Tile-matching the tremendous irregular sparsity](https://papers.nips.cc/paper/7666-tetris-tile-matching-the-tremendous-irregular-sparsity)  [NIPS '18]  | block-wise weight sparsity

- [Hybrid pruning: Thinner sparse networks for fast inference on edge devices](http://arxiv.org/abs/1811.00482)    [arXiv '18]

- [Deep gradient compression: Reducing the communication bandwidth for distributed training](https://arxiv.org/abs/1712.01887)  [arXiv  '17]

- [Learning efficient convolutional networks through network slimming](https://arxiv.org/abs/1708.06519)   [ICCV '17] | L1 regularization on BN | time 

- [Exploring sparsity in recurrent neural networks](http://arxiv.org/abs/1704.05119)  [arXiv  '17]

- [Less Is More: Towards Compact CNNs](http://users.umiacs.umd.edu/~hzhou/paper/zhou_ECCV2016.pdf)  [ECCV  '16] | 优化l1 l2

- [Learning Structured Sparsity in Deep Neural Networks](https://arxiv.org/pdf/1608.03665.pdf)   [arXiv  '16]  | **首篇引入正则化**

# 4 Random and Rethinking

> 有采用各种剪枝方法的就有和这些剪枝方法对着干的。

- [Rethinking the Value of Network Pruning](https://zhuanlan.zhihu.com/write)  [ICLR '19]

  >  the structure of the pruned model is more important than the inherited “important” weights

- [Recovering from Random Pruning: On the Plasticity of Deep Convolutional Neural Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1801.10447.pdf) [arXiv  '18]

  >  另外，在文献[13]中作者指出，裁剪之后仍能保 持模型性能并不是归功于所选择的特定裁剪标准， 而是由于深层神经网络的固有可塑性，这种可塑性 使得网络在精调后能够恢复裁剪造成的精度损失， 因此随机裁剪也可以达到在保证精度的同时极大地 压缩网络的目标。

- [Pruning from Scratch](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1909.12579)  [arXiv  '19]

# 5 Others and searching prunings

> 包括但不局限于：Meta-learning 、NAS 、

- [DMCP: Differentiable Markov Channel Pruning for Neural Networks](https://arxiv.org/abs/2005.03354) | [code](https://github.com/zx55/dmcp) [CVPR  '20] | NAS + Channel Pruning

- [Meta-Learning with Network Pruning](https://arxiv.org/abs/2007.03219)   [ECCV '20]

- [Comparing Rewinding and Fine-tuning in Neural Network Pruning](https://openreview.net/forum?id=S1gSj0NKvB)  | [code](https://github.com/lottery-ticket/rewinding-iclr20-public)  [ICLR '20]

- [DHP: Differentiable Meta Pruning via HyperNetworks](https://arxiv.org/abs/2003.13683)   | [code star 27](https://github.com/ofsoundof/dhp)  [ECCV '20]

- [A Signal Propagation Perspective for Pruning Neural Networks at Initialization](https://openreview.net/forum?id=HJeTo2VFwH)   [ICLR  '20]

- [Metapruning:  Meta learning for automatic neural network channel pruning](https://arxiv.org/abs/1903.10258)  [ICCV '19] | first automatic

- Network pruning via transformable architecture search.   2019.

- Autocompress: An automatic dnn structured pruning framework for ultra-high compression rates  2019

- ADC: automated deep compression and acceleration with reinforcement learning  2019 | 强化学习

- [Autoprune: Automatic network pruning by regularizing auxiliary parameters](http://papers.nips.cc/paper/9521-autoprune-automatic-network-pruning-by-regularizing-auxiliary-parameters.pdf)   [NIPS  '19]

- [AutoPruner: An end-to-end trainable filter pruning method for efficient deep model inference](http://arxiv.org/abs/1805.08941)   [arXiv '18]

- Constraint-aware deep neural network compression  [ECCV  '18]

- [AMC: Automl for model compression and acceleration on mobile devices](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yihui_He_AMC_Automated_Model_ECCV_2018_paper.pdf) [ECCV  '18]  | 强化学习

  > 将强化学习引入剪枝

- Auto-balanced filter pruning for efficient convolutional neural networks  [AAAI '18]

- Runtime neural pruning  [NIPS  '17]
