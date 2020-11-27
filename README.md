# LADMM-Net: An Unrolled Deep Network For Spectral Image Fusion From Compressive Data

Juan Marcos Ramírez, José Ignacio Martínez Torre, [Henry Arguello](http://hdspgroup.com/ "HDSP's Homepage")

## Abstract

The hyperspectral (MS) and multispectral (MS) image fusion is a signal processing task that aims at recovering a high-resolution spectral image from a low-spatial-resolution HS image and low-spectral-resolution MS image. In general, image fusion techniques are applied to data sets captured under the Nyquist-Shannon sampling theorem demanding expensive storing and processing capabilities of the acquisition systems. In this regard, coded aperture snapshot spectral imaging (CASSI) systems have emerged as an alternative acquisition framework based on the compressive sensing theory that obtains the relevant information of spectral images using a reduced number of encoded projections. In this work, a deep learning architecture is proposed for solving the spectral image fusion problem from HS and MS compressive measurements. More precisely, dubbed LADMM-Net, casts each iteration of the linearized version of the alternating direction method of multipliers (ADMM) into a processing layer that relied on a convolutional neural network. The linearized approach of the ADMM algorithm allows the reduction of the computation complexity related to the target variable estimate. Additionally, network parameters are learned using an end-to-end training strategy, improving thus, the performance of the proposed approach. The performance of the proposed spectral image fusion technique is evaluated on two spectral image databases. Furthermore, the proposed approach is tested on CASSI measurements captured at the laboratory. Extensive simulations show that the proposed method outperforms the state-of-the-art techniques that fuse spectral images from compressive measurements.
