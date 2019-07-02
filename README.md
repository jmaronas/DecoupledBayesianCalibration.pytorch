# BayesianDeepModelCalibration.pytorch
Code for reproducing some of the experiments of the Preprint Work named Calibration of Deep Probabilistic Models with Decoupled Bayesian Neural Networks. Follow the instructions.

## Important

As PyTorch is a recent software, the experiments performed in this work have been done over two different releases, as the second one provide lots of advantages over the initial one. Thus, to ensure reproducibility we provide code and details on the models in separate folders. 

This code is uploaded just to justify the results provided in the work, and not for a general purpose application. This means that, for instance, code is only prepared to run in GPU and python2.7. Moreover, not all the code has been made to be able to manage any topology. We use some code to make an initial check that was use on some experiments. Once we saw that our hypothesis worked we did more a efficient implementation. Feel free to modify it. 

In the work we report more than 60 experiments. We uploaded some data files to google drive, but not all of them. Feel free to send an email to [elmaronias@gmail.com](elmaronias@gmail.com) requesting the logits for a particular experiment.

Again, we will not update this code to python3 (python 2 is deprecated starting on 2020). This is because the results might be affected if the way of generating random numbers changes. This is one of the main reasons for uploading the unoptimized code, as the results using the optimized version changed and we did not want to redoo many of the experiments. 

## Software and Hardware Requirements

To ensure reproducibility we provide the hardware and software used.

Software: pytorch version 0.3.1, python 2.7, cuda 8, ubuntu 16 (stable version) ; pytorch version 0.4.0, python2.7, cuda9.1, ubuntu 16 (stable version)

Hardware (GPU): Nvidia 1080 and Nvidia TitanXp. Intel based platforms

You will need awk, wget, grep and some typical unix tools to run the provided examples, for instance.

## Installation

To install pytorch (in a virtual enviroment) follow the next steps:

Pytorch 0.3.1:

   ```
   virtualenv -p python2.7 /tmp/0.3.1_cuda8_pytorch
   source /tmp/0.3.1_cuda8_pytorch/bin/activate
   pip install  http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl
   ```

Pytorch 0.4.0

  ```
  virtualenv -p python2.7 /tmp/0.4.0_cuda9.1_pytorch
  source /tmp/0.4.0_cuda9.1_pytorch/bin/activate
  pip install  https://download.pytorch.org/whl/cu91/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl
  pip install numpy # by january 2019, this installation does not install the requirements. This did not happened before (for 0.3.1 it install everything). We remark this explicitly because  pip used to install  all the requirements when this version was installed, but now it does not. This means that this can change in the future. You can use conda if you prefer. 
  ```

Anyway, in the two examples that we provide (continue reading) the example installs the enviroment for you

## Dataset

You can download the logits for some experiments from [here](https://drive.google.com/drive/folders/1uJ06kNMDRGcZ-6xjxHYtqwC6ea4eSGcl?usp=sharing). Place the downloaded data folder with the provided name in a directory, as example, /tmp/data/

## Experiments

The experiments for the standard BNN are placed in folder standard_BNN. The experiments for the BNN with LocalReparameterization are placed in folder LR_BNN. Follow the instructions inside these folders. 

## Aknowledgment

This work is partially supported by Nvidia-Corporation as we used two gpus from their gpu donation program. Thanks!
