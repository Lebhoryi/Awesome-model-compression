[TOC]

# 2001.08514

基于结构和非结构化分类

## Unstructured method

[1] Jose M Alvarez and Mathieu Salzmann. Learning the number of neurons in deep networks. In Advances in Neural Information Processing Systems (NeurIPS), pages 2270–2278, 2016.

[2] Xin Dong, Shangyu Chen, and Sinno Pan. Learning to prune deep neural networks via layerwise optimal brain surgeon. In Advances in Neural Information Processing Systems (NeurIPS), pages 4857–4867, 2017.

[5] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations (ICLR), 2019.

[8] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in Neural Information Processing Systems (NeurIPS), pages 1135–1143, 2015.

[26] Zhenhua Liu, Jizheng Xu, Xiulian Peng, and Ruiqin Xiong. Frequency-domain dynamic pruning for convolutional neural networks. In Advances in Neural Information Processing Systems (NeurIPS), pages 1043–1053, 2018.

## Structured method

### 1.  Regularization-based pruning

[13] Zehao Huang and Naiyan Wang. Data-driven sparse structure selection for deep neural networks. In European Conference on Computer Vision (ECCV), pages 304–320, 2018.

[24] Shaohui Lin, Rongrong Ji, Chenqian Yan, Baochang Zhang, Liujuan Cao, Qixiang Ye, Feiyue Huang, and David Doermann. Towards optimal structured cnn pruning via generative adversarial learning. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2790–2799, 2019.

[27] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In IEEE International Conference on Computer Vision (ICCV), pages 2736–2744, 2017.

[40] Chenglong Zhao, Bingbing Ni, Jian Zhang, Qiwei Zhao, Wenjun Zhang, and Qi Tian. Variational convolutional neural network pruning. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2780–2789, 2019.

### 2. Property-based pruning

[12] Hengyuan Hu, Rui Peng, Yu-Wing Tai, and Chi-Keung Tang. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. arXiv preprint arXiv:1607.03250, 2016.

[20] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In International Conference on Learning Representations (ICLR), 2017.

[38] Ruichi Yu, Ang Li, Chun-Fu Chen, Jui-Hsin Lai, Vlad I Morariu, Xintong Han, Mingfei Gao, Ching-Yung Lin, and Larry S Davis. Nisp: Pruning networks using neuron importance score propagation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 9194–9203, 2018.

### 3. Optimization-based pruning

[28] Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In IEEE International Conference on Computer Vision (ICCV), pages 5058–5066, 2017.

[11] Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In IEEE International Conference on Computer Vision (ICCV), pages 1389–1397, 2017.

and 该篇

# 2005.03354

## 1. hard pruned 

移除通道

- [9] Hengyuan Hu, Rui Peng, Yu-Wing Tai, and Chi-Keung Tang. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. arXiv preprint arXiv:1607.03250, 2016. 2, 4 
- [10] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710, 2016. 1, 2, 4
- [16] Pavlo Molchanov, Stephen Tyree, Tero Karras, Timo Aila, and Jan Kautz. Pruning convolutional neural networks for resource efficient inference. arXiv preprint arXiv:1611.06440, 2016. 1, 2

## 2. soft pruned

[3] Xiaohan Ding, Guiguang Ding, Yuchen Guo, Jungong Han, and Chenggang Yan. Approximated oracle filter pruning for destructive CNN width optimization. CoRR, abs/1905.04748, 2019. 2

[6] Yang He, Xuanyi Dong, Guoliang Kang, Yanwei Fu, and Yi Yang. Progressive deep neural networks acceleration via soft filter pruning. arXiv preprint arXiv:1808.07471, 2018. 1, 2, 8

[12] Shaohui Lin, Rongrong Ji, Chenqian Yan, Baochang Zhang, Liujuan Cao, Qixiang Ye, Feiyue Huang, and David S. Doermann. Towards optimal structured CNN pruning via generative adversarial learning. CoRR, abs/1903.09291, 2019. 2

and 该篇

# 2003.08935

## 1. nonstructural pruning

[11] Song Han, Huizi Mao, and William J Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding. In *Proc. ICLR*,

\2015. 2

 [12] Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Proc. NeurIPS, pages 1135–1143, 2015. 2

 [32] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In Proc. ICCV, pages 2736–2744, 2017. 3, 4

## 2. Structural pruning

 [1] Jose M Alvarez and Mathieu Salzmann. Learning the number of neurons in deep networks. In Proce. NeurIPS, pages 2270–2278, 2016. 3

 [46] Wei Wen, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Learning structured sparsity in deep neural networks. In Proc. NeurIPS, pages 2074–2082, 2016. 3

 [57] Hao Zhou, Jose M Alvarez, and Fatih Porikli. Less is more: Towards compact cnns. In Proc. ECCV, pages 662–677. Springer, 2016. 3

 [15] Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, and Yi Yang. Filter pruning via geometric median for deep convolutional neural networks acceleration. In Proc. CVPR, pages 4340– 4349, 2019. 3, 7, 8

 [16] Yihui He, Xiangyu Zhang, and Jian Sun. Channel pruning for accelerating very deep neural networks. In Proc. ICCV, pages 1389–1397, 2017. 1, 3, 13

 [53] Dejiao Zhang, Haozhu Wang, Mario Figueiredo, and Laura Balzano. Learning to share: Simultaneous parameter tying and sparsification in deep learning. In Proc. ICLR, 2018. 3 [26] Jiashi Li, Qi Qi, Jingyu Wang, Ce Ge, Yujian Li, Zhangzhang Yue, and Haifeng Sun. OICSR: Out-in-channel sparsity regularization for compact deep neural networks. In Proc. CVPR, pages 7046–7055, 2019. 3, 4

 [43] Amirsina Torfi, Rouzbeh A Shirvani, Sobhan Soleymani, and Naser M Nasrabadi. GASL: Guided attention for sparsity learning in deep neural networks. arXiv preprint arXiv:1901.01939, 2019. 3

 [19] Zehao Huang and Naiyan Wang. Data-driven sparse structure selection for deep neural networks. In Proc. ECCV, pages 304–320, 2018. 3, 4, 7, 8, 12, 14

 [32] Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, and Changshui Zhang. Learning efficient convolutional networks through network slimming. In Proc. ICCV, pages 2736–2744, 2017. 3, 4

# AutoPrune: Automatic Network Pruning by Regularizing Auxiliary Parameters
## 1. Unstructural pruning

- Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for effificient neural network. In *Advances in neural information processing systems*, pages 1135–1143, 2015.

- Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for effificient dnns. In *Advances In Neural* *Information Processing Systems*, pages 1379–1387, 2016.
- Xin Dong, Shangyu Chen, and Sinno Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. In *Advances in Neural Information Processing Systems*, pages 4857–4867, 2017
- Alireza Aghasi, Afshin Abdi, Nam Nguyen, and Justin Romberg. Net-trim: Convex pruning of deep neural networks with performance guarantee. In *Advances in Neural Information Processing Systems*, pages 3177–3186, 2017.
- Enzo Tartaglione, Skjalg Lepsøy, Attilio Fiandrotti, and Gianluca Francini. Learning sparse neural networks via sensitivity-driven regularization. In *Advances in Neural Information Processing Systems*, pages 3878–3888, 2018.



- Yann LeCun, John S Denker, and Sara A Solla. Optimal brain damage. In *Advances in neural information processing systems*, pages 598–605, 1990.

-  Song Han, Jeff Pool, John Tran, and William Dally. Learning both weights and connections for efficient neural network. In Advances in neural information processing systems, pages 1135–1143, 2015.
-  Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. In Advances In Neural Information Processing Systems, pages 1379–1387, 2016.
-  Hengyuan Hu, Rui Peng, Yu-Wing Tai, and Chi-Keung Tang. Network trimming: A data-driven neuron pruning approach towards efficient deep architectures. arXiv preprint arXiv:1607.03250, 2016.
-  Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. Pruning filters for efficient convnets. In International Conference on Learning Representations, 2017.

## 2. Structural pruning

-  Ariel Gordon, Elad Eban, Ofir Nachum, Bo Chen, Hao Wu, Tien-Ju Yang, and Edward Choi. Morphnet: Fast & simple resource-constrained structure learning of deep networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1586–1595, 2018.
- Xiaotian Zhu, Wengang Zhou, and Houqiang Li. Improving deep neural network sparsity through decorrelation regularization. In *IJCAI*, pages 3264–3270, 2018.
-  Zhuangwei Zhuang, Mingkui Tan, Bohan Zhuang, Jing Liu, Yong Guo, Qingyao Wu, Junzhou Huang, and Jinhui Zhu. Discrimination-aware channel pruning for deep neural networks. In Advances in Neural Information Processing Systems, pages 883–894, 2018.
- Aidan N Gomez, Ivan Zhang, Kevin Swersky, Yarin Gal, and Geoffrey E Hinton. Learning sparse networks using targeted dropout. *arXiv preprint arXiv:1905.13678*, 2019.
-  Hanxiao Liu, Karen Simonyan, and Yiming Yang. DARTS: Differentiable architecture search. In *International* *Conference on Learning Representations*, 2019.
- Zhuang Liu, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. Rethinking the value of network pruning. In *International Conference on Learning Representations*, 2019.
-  Jiahui Yu and Thomas Huang. Network slimming by slimmable networks: Towards one-shot architecture search for channel numbers. arXiv preprint arXiv:1903.11728, 2019.
-  Jiahui Yu, Linjie Yang, Ning Xu, Jianchao Yang, and Thomas Huang. Slimmable neural networks. arXiv preprint arXiv:1812.08928, 2018.

# 1905.10138

## Quantized with pruning

- [viterbi-based compression]

   [1] Daehyun Ahn, Dongsoo Lee, Taesu Kim, and Jae-Joon Kim. Double Viterbi: Weight encoding for high compression ratio and fast on-chip reconstruction for deep neural network. In International Conference on Learning Representations (ICLR), 2019. 1, 3, 6

   [19] Dongsoo Lee, Daehyun Ahn, Taesu Kim, Pierce I. Chuang, and Jae-Joon Kim. Viterbi-based pruning for sparse matrix with fixed and high index compression ratio. In International Conference on Learning Representations (ICLR), 2018. 1, 3, 6

- [Deep compression]

   [10] Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding. In International Conference on Learning Representations (ICLR), 2016. 1, 2

- [ternary weight networks (TWN)]

   [23] Fengfu Li and Bin Liu. Ternary weight networks. arXiv:1605.04711, 2016. 1, 6

- [trained ternary quantization (TTQ)]

  [36] Chenzhuo Zhu, Song Han, Huizi Mao, and William J. Dally. Trained ternary quantization. In *International Conference on* *Learning Representations (ICLR)*, 2017. 1, 6

# LFPC

![](https://gitee.com/lebhoryi/PicGoPictureBed/raw/master/img/20201021101614.png)

## Weight pruning

[2] M. A. Carreira-Perpinan and Y. Idelbayev. “learning- ´ compression” algorithms for neural net pruning. In CVPR, 2018. 2

[6] X. Dong, S. Chen, and S. Pan. Learning to prune deep neural networks via layer-wise optimal brain surgeon. In NeurIPS, 2017. 2

[12] Y. Guo, A. Yao, and Y. Chen. Dynamic network surgery for efficient DNNs. In NeurIPS, 2016. 2

[13] S. Han, H. Mao, and W. J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and huffman coding. In ICLR, 2015. 2

[14] S. Han, J. Pool, J. Tran, and W. Dally. Learning both weights and connections for efficient neural network. In NeurIPS, 2015. 1, 2

[48] F. Tung and G. Mori. Clip-q: Deep network compression learning by in-parallel pruning-quantization. In CVPR, 2018. 2

[56] T. Zhang, S. Ye, K. Zhang, J. Tang, W. Wen, M. Fardad, and Y. Wang. A systematic dnn weight pruning framework using alternating direction method of multipliers. ECCV, 2018. 2

## Filter pruning

### Weight-based criteria

[17] Y. He, X. Dong, G. Kang, Y. Fu, C. Yan, and Y. Yang. Asymptotic soft filter pruning for deep convolutional neural networks. IEEE Transactions on Cybernetics, pages 1–11, 2019. 2

[18] Y. He, G. Kang, X. Dong, Y. Fu, and Y. Yang. Soft filter pruning for accelerating deep convolutional neural networks. In IJCAI, 2018. 2, 5, 6, 7

[20] Y. He, P. Liu, Z. Wang, and Y. Yang. Pruning filter via geometric median for deep convolutional neural networks acceleration. In CVPR, 2019. 1, 2, 3, 4, 5, 6, 7, 8

[21] Y. He, P. Liu, L. Zhu, and Y. Yang. Meta filter pruning to accelerate deep convolutional neural networks. arXiv preprint arXiv:1904.03961, 2019. 2

[27] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf. Pruning filters for efficient ConvNets. In ICLR, 2017. 1, 2, 3, 4, 5, 6, 7

[50] X. Wang, Z. Zheng, Y. He, F. Yan, Z. Zeng, and Y. Yang. Progressive local filter pruning for image retrieval acceleration. arXiv preprint arXiv:2001.08878, 2020. 2

[51] J. Ye, X. Lu, Z. Lin, and J. Z. Wang. Rethinking the smaller-norm-less-informative assumption in channel pruning of convolution layers. In ICLR, 2018. 2

## Activation-based criteria

[10] A. Dubey, M. Chatterjee, and N. Ahuja. Coreset-based neural network compression. arXiv preprint arXiv:1807.09810, 2018. 2

[19] Y. He, J. Lin, Z. Liu, H. Wang, L.-J. Li, and S. Han. Amc: Automl for model compression and acceleration on mobile devices. In ECCV, 2018. 2, 3, 6

[22] Y. He, X. Zhang, and J. Sun. Channel pruning for accelerating very deep neural networks. In ICCV, 2017. 2, 5, 6, 7, 8

[23] Q. Huang, K. Zhou, S. You, and U. Neumann. Learning to prune filters in convolutional neural networks. In WACV, 2018. 2, 3

[24] Z. Huang and N. Wang. Data-driven sparse structure selection for deep neural networks. In ECCV, 2018. 2, 6, 7

[29] S. Lin, R. Ji, Y. Li, Y. Wu, F. Huang, and B. Zhang. Accelerating convolutional networks via global & dynamic filter pruning. In IJCAI, 2018. 2

[30] S. Lin, R. Ji, C. Yan, B. Zhang, L. Cao, Q. Ye, F. Huang, and D. Doermann. Towards optimal structured cnn pruning via generative adversarial learning. In CVPR, 2019. 2, 6

[36] J.-H. Luo, J. Wu, and W. Lin. ThiNet: A filter level pruning method for deep neural network compression. In ICCV, 2017. 2, 5, 6, 7, 8

[38] P. Molchanov, A. Mallya, S. Tyree, I. Frosio, and J. Kautz. Importance estimation for neural network pruning. In CVPR, 2019. 2, 6, 7

[39] P. Molchanov, S. Tyree, T. Karras, T. Aila, and J. Kautz. Pruning convolutional neural networks for resource efficient transfer learning. In ICLR, 2017. 2

[41] H. Peng, J. Wu, S. Chen, and J. Huang. Collaborative channel pruning for deep networks. In ICML, 2019. 2

[45] X. Suau, L. Zappella, V. Palakkode, and N. Apostoloff. Principal filter analysis for guided network compression. arXiv preprint arXiv:1807.10585, 2018. 2

[54] R. Yu, A. Li, C.-F. Chen, J.-H. Lai, V. I. Morariu, X. Han, M. Gao, C.-Y. Lin, and L. S. Davis. NISP: Pruning networks using neuron importance score propagation. In CVPR, 2018. 2, 3, 6, 7

[57] Z. Zhuang, M. Tan, B. Zhuang, J. Liu, Y. Guo, Q. Wu, J. Huang, and J. Zhu. Discrimination-aware channel pruning for deep neural networks. In NeurIPS, 2018. 2

## Greedy and one-shot Pruning

1. Greedy

   [53] Z. You, K. Yan, J. Ye, M. Ma, and P. Wang. Gate decorator: Global filter pruning method for accelerating deep convolutional neural networks. arXiv preprint arXiv:1909.08174, 2019. 2, 3

   [5] X. Ding, G. Ding, Y. Guo, J. Han, and C. Yan. Approximated oracle filter pruning for destructive cnn width optimization. In ICML, 2019. 2, 3

2. One-shot

   [20] Y. He, P. Liu, Z. Wang, and Y. Yang. Pruning filter via geometric median for deep convolutional neural networks acceleration. In CVPR, 2019. 1, 2, 3, 4, 5, 6, 7, 8

   [27] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf. Pruning filters for efficient ConvNets. In ICLR, 2017. 1, 2, 3, 4, 5, 6, 7

## Other prunings and searching

1. 强化学习

   [19] Y. He, J. Lin, Z. Liu, H. Wang, L.-J. Li, and S. Han. Amc: Automl for model compression and acceleration on mobile devices. In ECCV, 2018. 2, 3, 6

   [23] Q. Huang, K. Zhou, S. You, and U. Neumann. Learning to prune filters in convolutional neural networks. In WACV, 2018. 2, 3

2. meta-learning

   [33] Z. Liu, H. Mu, X. Zhang, Z. Guo, X. Yang, T. K.-T. Cheng, and J. Sun. Metapruning: Meta learning for automatic neural network channel pruning. arXiv preprint arXiv:1903.10258, 2019. 3

3. NAS

   [58] B. Zoph and Q. V. Le. Neural architecture search with reinforcement learning. In ICLR, 2017. 3

   [31] H. Liu, K. Simonyan, and Y. Yang. Darts: Differentiable architecture search. In ICLR, 2019. 3

4. Autoaugment

   [3] E. D. Cubuk, B. Zoph, D. Mane, V. Vasudevan, and Q. V. Le. Autoaugment: Learning augmentation strategies from data. In CVPR, 2019. 3