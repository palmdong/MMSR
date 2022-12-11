# MMSR
PyTorch implementation of "Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution"  
[[ECCV2022 paper](https://link.springer.com/chapter/10.1007/978-3-031-19800-7_1), [supp](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790001-supp.pdf)] [[arXiv](https://arxiv.org/abs/2207.09156)]


## Updates
**[2022/07/04]** We thank anonymous reviewers from ECCV2022 and CVPR2022 for their suggestions to our paper. See you in Tel-Aviv!  
**[2022/07/19]** Paper is available on [arXiv](https://arxiv.org/abs/2207.09156).  
**[2022/07/20]** Code was uploaded. Our code was built on the [repository](https://github.com/prs-eth/PixTransform) of [P2P (ICCV2019)](https://arxiv.org/abs/1904.01501), we thank the authors for their efforts.   


## Abstract
Self-supervised cross-modal super-resolution (SR) can overcome the difficulty of acquiring paired training data, but is challenging because only low-resolution (LR) source and high-resolution (HR) guide images from different modalities are available. 
Existing methods utilize pseudo or weak supervision in LR space and thus deliver results that are blurry or not faithful to the source modality. 
To address this issue, we present a mutual modulation SR (MMSR) model, which tackles the task by a mutual modulation strategy, including a source-to-guide modulation and a guide-to-source modulation. 
In these modulations, we develop cross-domain adaptive filters to fully exploit cross-modal spatial dependency and help induce the source to emulate the resolution of the guide and induce the guide to mimic the modality characteristics of the source. 
Moreover, we adopt a cycle consistency constraint to train MMSR in a fully self-supervised manner. 
Experiments on various tasks demonstrate the state-of-the-art performance of our MMSR.


<p align="center"> <img src="figs/result_example.png" width="68%"> </p>


## Quick Start
Check the Jupyter Notebook file [inference.ipynb](https://github.com/palmdong/MMSR/blob/main/inference.ipynb).  
For any question or discussion, reach Xiaoyu Dong at dong@ms.k.u-tokyo.ac.jp.


## Citation
```
@InProceedings{Dong2022MMSR,
  author    = {Dong, Xiaoyu and Yokoya, Naoto and Wang, Longguang and Uezato, Tatsumi},
  title     = {Learning Mutual Modulation for Self-Supervised Cross-Modal Super-Resolution},
  booktitle = {ECCV},
  year      = {2022}
}
```
