# BF-design-with-DL
This is the simulation code for the paper "**Beamforming Design for Large-Scale Antenna Arrays Using Deep Learning**".  This paper is published on IEEE Wireless Communication Letters. 

IEEE link: [https://ieeexplore.ieee.org/document/8847377/](https://ieeexplore.ieee.org/document/8847377/)

Arxiv link: [https://arxiv.org/abs/1904.03657](https://arxiv.org/abs/1904.03657)

## Results
After fork the repo and download the corresponding data sets and trained models, the following performance results can be easily reproduced. (the python codes is only for the blue cerves, and compared cerves should be plot via Matlab codes)

## How to use:
***
* run the train.py to train the model 
* run the test.py to test the trained model 
* **Due to the space limitation of github, we provide two tiny training and testing data sets only for running the example.**

## Data sets and trained models
***
* Thanks for the authors of [[10]](http://oa.ee.tsinghua.edu.cn/dailinglong/publications/paper/Reliable%20beamspace%20channel%20estimation%20for%20millimeter-wave%20massive%20MIMO%20systems%20with%20lens%20antenna%20array.pdf), the simulation codes of the channel is provided [here](http://oa.ee.tsinghua.edu.cn/dailinglong/publications/publications.html). you 
can use the [direct URL](http://oa.ee.tsinghua.edu.cn/dailinglong/publications/code/Reliable%20beamspace%20channel%20estimation%20for%20millimeter-wave%20massive%20MIMO%20systems%20with%20lens%20antenna%20array.zip) to download it.
* The channel estimation codes can be referred to the website of the first author of [[2]](https://ieeexplore.ieee.org/document/6847111).
* We have provided the data sets in our [google drive](https://drive.google.com/open?id=1nSk9TftoCMA5iRUqC9GSK5g5a67Q8FRG), which can be directly used in our .py files.
* Trained weights, corresponding to the shown results in the paper, is also provided in the [google drive](https://drive.google.com/open?id=1nSk9TftoCMA5iRUqC9GSK5g5a67Q8FRG).

## Samples Generation
For samples generation. Please kindly refer to the gen_samples.m for details. The codes are based on the work [2].

## End
***
* Matlab codes of compared algorithms [4，5] can be referred to [this repo](https://github.com/TianLin0509/Hybrid-Beamforming-for-Millimeter-Wave-Systems-Using-the-MMSE-Criterion). In addition, if you are interested in traditional HBF algorithms, you can kindly refer to our previous work [Hybrid Beamforming for Millimeter Wave Systems Using the MMSE Criterion](https://arxiv.org/abs/1902.08343?context=cs.IT), we also provide specific Matlab codes for your reproduction. You can refer to [this repo](https://github.com/TianLin0509/Hybrid-Beamforming-for-Millimeter-Wave-Systems-Using-the-MMSE-Criterion). 
* If you have any questions, you can contact the author via ```lint17@fudan.edu.cn``` for help.
* Hopefully you can **star** or **fork** this repo if it is helpful. Thank you for your support in advance.
