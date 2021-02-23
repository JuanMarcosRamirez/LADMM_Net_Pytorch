# LADMM-Net: An Unrolled Deep Network For Spectral Image Fusion From Compressive Data

[Juan Marcos Ramírez](https://juanmarcosramirez.github.io/ "Juan's Website"), José Ignacio Martínez Torre, [Henry Arguello](http://hdspgroup.com/ "HDSP's Homepage")

## Abstract

Hyperspectral (HS) and multispectral (MS) image fusion aims at estimating a high-resolution spectral image from a low-spatial-resolution HS image and a low-spectral-resolution MS image. Fusion methods typically use HS and MS images acquired by scanning techniques that require a huge amount of measurements to represent the target spectral field. In this regard, compressive spectral imaging (CSI) has emerged as an acquisition framework based on the compressive sampling theory that captures the relevant information of spectral images using a reduced number of encoded snapshots. Recently, various spectral image fusion methods from multi-sensor CSI measurements have been proposed. Nevertheless, these methods exhibit high running times, and they also face the drawback of choosing a predefined transform to represent the fused images. In this work, a deep learning architecture under the algorithm unrolling approach is proposed for solving the fusion problem from HS and MS compressive measurements. More precisely, this architecture, dubbed LADMM-Net, casts each iteration of a linearized version of the alternating direction method of multipliers (ADMM) into a network-based processing layer whose instances are concatenated to form a deep network. The linearized approach of the ADMM algorithm leads to estimate the target variable without resorting to expensive matrix inversions. This approach also estimates the image high-frequency component included in both the auxiliary variable and the Lagrange multiplier by using a network-based transform. Network parameters are learned using an end-to-end training strategy, improving thus, the performance of the proposed approach. The performance of the proposed fusion technique is evaluated on two spectral image databases and one real dataset captured at the laboratory. Extensive simulations show that the proposed method outperforms the state-of-the-art approaches that fuse spectral images from compressive data.


![Demo image](https://github.com/JuanMarcosRamirez/LADMM_Net_Pytorch/blob/master/images/architecture.jpg?raw=true "Demo houston")

![Demo image](https://github.com/JuanMarcosRamirez/LADMM_Net_Pytorch/blob/master/images/Fusion1.jpg?raw=true "Cork Wall")

![Demo image](https://github.com/JuanMarcosRamirez/LADMM_Net_Pytorch/blob/master/images/Fusion2.jpg?raw=true "Door")

![Demo image](https://github.com/JuanMarcosRamirez/LADMM_Net_Pytorch/blob/master/images/Fusion3.jpg?raw=true "House")


## Dataset

To reproduce the paper results, please download spectral images and coded aperture patterns from this [Google Drive.](https://drive.google.com/drive/folders/1cMRJnMuyd9zdi0vQxkxwCFJUX8HhfZYj?usp=sharing "Training dataset link")

## CS Reconstruction

![Demo image](https://github.com/JuanMarcosRamirez/LADMM_Net_Pytorch/blob/master/images/Hector.png?raw=true "Baby image")

![Demo image](https://github.com/JuanMarcosRamirez/LADMM_Net_Pytorch/blob/master/images/Boy.png?raw=true "Boy image")

### Platform

* Ubuntu 18.04 Operating System.

### License

This code package is licensed under the GNU GENERAL PUBLIC LICENSE (version 3) - see the [LICENSE](LICENSE) file for details.

### Contact

[Juan Marcos Ramirez](juanmarcos.ramirez@ujrc.es)

### Date

February 15, 2021

### Acknowledgements

This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 754382, GOT ENERGY TALENT. The content of this article does not reflect the official opinion of the European Union. Responsibility for the information and views expressed herein lies entirely with the authors.

