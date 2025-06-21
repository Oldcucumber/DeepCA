# DeepCA: Deep Learning-based 3D Coronary Artery Tree Reconstruction from Two 2D Non-simultaneous X-ray Angiography Projections

# 1. Overview

This is the official code repository for the [DeepCA](https://arxiv.org/abs/2407.14616) paper by Yiying Wang, Abhirup Banerjee, Robin P. Choudhury and Vicente Grau, which has been early accepted to the **WACV 2025 (Oral) (WACV Broadening Participation Scholarship - Travel Award: Top 2\%) (early acceptance rate: 12.1%)**.

## Citation

If you find the code are useful, please consider citing the paper.

```
@article{wang2024deep,
  title={Deep Learning-based 3D Coronary Tree Reconstruction from Two 2D Non-simultaneous X-ray Angiography Projections},
  author={Wang, Yiying and Banerjee, Abhirup and Choudhury, Robin P and Grau, Vicente},
  journal={arXiv preprint arXiv:2407.14616},
  year={2024}
}
```

# 2. Introduction

Cardiovascular diseases (CVDs) are the most common cause of death worldwide. Invasive x-ray coronary angiography (ICA) is one of the most important imaging modalities for the diagnosis of CVDs. ICA typically acquires only two 2D projections, which makes the 3D geometry of coronary vessels difficult to interpret, thus requiring 3D coronary tree reconstruction from two projections. State-of-the-art approaches require significant manual interactions and cannot correct the non-rigid cardiac and respiratory motions between non-simultaneous projections. In this study, we propose a novel deep learning pipeline called DeepCA. We leverage the Wasserstein conditional generative adversarial network with gradient penalty, latent convolutional transformer layers, and a dynamic snake convolutional critic to implicitly compensate for the non-rigid motion and provide 3D coronary artery tree reconstruction. Through simulating projections from coronary computed tomography angiography (CCTA), we achieve the generalisation of 3D coronary tree reconstruction on real non-simultaneous ICA projections. We incorporate an application-specific evaluation metric to validate our proposed model on both a CCTA dataset and a real ICA dataset, together with Chamfer L1 distance. The results demonstrate promising performance of our DeepCA model in vessel topology preservation, recovery of missing features, and generalisation ability to real ICA data. To the best of our knowledge, this is the first study that leverages deep learning to achieve 3D coronary tree reconstruction from two real non-simultaneous x-ray angiographic projections. 

## Our Proposed Pipeline

Our proposed DeepCA method consists of two blocks: a data preprocessing block and a 3D reconstruction with motion compensation block. In the data preprocessing block, we generate two simulated ICA projections based on 3D CCTA data, with simulated motion on the second projection plane, and then apply backprojection on them to produce the input to the DeepCA model at the next block. In the 3D reconstruction with motion compensation block, we map the 3D backprojection input to the CCTA data for 3D coronary artery tree reconstruction via training a deep neural network, implicitly compensating for any motion.

<p align="center">
  <img src="https://github.com/WangStephen/DeepCA/blob/main/img/workflow.png">
</p>

## Our Proposed Model

Our DeepCA model architecture is based on the Wasserstein conditional generative adversarial network with gradient penalty, latent convolutional transformer layers, and a dynamic snake convolutional critic. Via mapping the input with non-aligned projections to 3D coronary tree data, most motion artifacts are corrected by our model. With the critic used, any residual uncorrected deformations are adjusted, while ensuring the connectedness of the coronary tree structures in the reconstructions and increasing the model's elastic generalisation capacity. So when generalising to real non-simultaneous ICA projections, the non-rigid motion is compensated implicitly.

<p align="center">
  <img src="https://github.com/WangStephen/DeepCA/blob/main/img/model.png">
</p>

# 3. Packages Requirement

This work requires following dependency packages:

```
python: 3.9.18
pytorch: 2.1.1
numpy: 1.23.5 
nibabel: 3.2.2
tigre 
```

# 4. Code Instructions

## Training Data Preparation

Our training data are based on the segmented CCTA data (label) from [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).

We first split the segmented CCTA data into RCA and LAD and then use the RCA data to simulate cone-beam forward projections. Please run the functions `CCTA_split` and `generate_deformed_projections_RCA` in `projections_simulation.py` to generate your simulated deformed projections from the segmented CCTA data.

## Model Training

Run the model to start training:

```
python ./DeepCA/train_models/train_model.py 
```

Once finished, the model trained on simulated CCTA projections can then be generalised to real clinical non-simultaneous ICA data for 3D coronary artery tree reconstruction. 

# 5. License

Please see [license](https://github.com/WangStephen/DeepCA/blob/main/LICENSE).

# 6. Acknowledgement

DeepCA model architectures are revised based on [Compact Transformers](https://github.com/SHI-Labs/Compact-Transformers) and [Dynamic Snake Convolution](https://github.com/YaoleiQi/DSCNet).

Cone-beam forward projection simulation is based on [TIGRE Toolbox](https://github.com/CERN/TIGRE).
