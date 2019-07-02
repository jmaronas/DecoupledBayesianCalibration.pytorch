# BNN without applying Local Reparameterization

We provide the basic information to reproduce the results. Refer to the main (anonymized for the moment) work to check which models you can run under this directory. Anyway, the code is ready to run any model on any dataset for instance if you want to check the accuracy degradation during training that appears if you want to train models on cars and birds. Execute python file.py --help for further information (for example on which kind of information or choices can be given to the python parser)

## Material provided

Code for running experiments for BNN

Code for running experiments of Temperature Scaling

Some bash utilities -> under the folder ./utilities/

Two version of code. One is an optimized version of the other

A folder with a particular example that does everything for you: download data, run temperature scaling... Just go inside this folder and execute ./example.sh

## Software required

This code is prepared to be used with pytorch version 0.3.1. Activate your virtual enviroment before running any of the files provided.

Use bash instead of dash. This is not a limitation but is preferable if you are going to use the bash scripts provided

## Replicate an example of the paper.

We provide an easy way to replicate one of the experiment of the paper. To do this you do not have to install anything, we do it for you. Just go inside the example folder and execute example.sh. This file will download pytorch, download a set of data, train temperature scaling and train a BNN. It will go slow because it uses the non-optimized version of the code. You have another example in LR_BNN folder which runs faster. You can follow the guidelines from this example to replicate any of the experiments provided. 

Guidelines (assuming you meet the hardware requirements and use linux):

     git clone https://github.com/2019submission/ICML2019.submission.git
     cd ./ICML2019.submission/standard_BNN/example
     ./example.sh

## Baseline Results

To obtain baseline calibration results (temperature scaling) run:

```
python main_plat_scaling.py --model_net [model_net] --data_dir [data_dir] --dataset [choosed_dataset] --epochs [epochs] --T_factor [T_factor]
```

### Code Params

model_net: name of the model that computs the logits, as example, densenet-121.

data_dir: location of your downloaded data, as example, /tmp/data/

choosed_dataset: which dataset you want to use, as example, cifar10.

T_factor: if provided calibrate test and validation set with the provided value. If not provided the code automatically optimize on validation to search for the T_factor that minimizes cross entropy. This should give you a unique value for T_factor as it is convex optimization problem. (In fact is convex if T scal multiplies instead of divides. However in these specific problems we always reach an optimum. If this that not happens to you in other kind of experiments just change the parameter to multiply instead of divide. We divide to follow GUO et al 2017 and make it easier to follow)

epochs: number of epochs for optimization (only used if T_factor is not provided).

Make any necessary changes in the code if you want to use other optimizer, change momentum, change initialization... In fact it should only affect the convergence but not the T_scal parameter.


## Training Variational Distribution (optimize the ELBO)

We did a set of initial experiments on CIFAR10 and SVHN. However, when we moved to CIFAR100 and due to the complexity of the models used for this task, we optimized the code. The models for ADIENCE, CIFAR10 and SVHN are runned under the "main_ELBO.py" script. The models for CIFAR100 are runned under the "main_ELBO_optimized.py" script. The first one only supports two hidden layers BNNs. Here I put an example using CIFAR100 however the arguments are the same for the other script.


```
python main_ELBO_optimized.py --model_net [model_net] --data_dir [data_dir] --dataset [choosed_dataset] --MC_samples [MC] --dkl_after_epoch [DAE] --dkl_scale_factor [DSF] --save_after [after_nepochs] --layer_dim [ldim] --n_layers [nlay] --epochs [epochs_used] --lr [lr_used] --batch  [batch_used] --anneal [anneal_used] --n_gpu [gpu_id] --folder_name [folder_name_provided]
```
### Code Params

model_net: name of the model that computed the logits, as example, densenet-121

data_dir: location of your downloaded data, as example, /tmp/data/

choosed_dataset: which dataset you want to use, as example, cifar10

MC: number of montecarlo samples to estimate the likelihood expectation under the variational distribution, as example, 30.

DAE: after which epoch we also optimize the DKL (known as warm up). If provide -1 it add the DKL term from the beginning.

DSF: factor to scale the DKL (\beta in the paper), as example 0.1

after_nepochs: after this number of epoch we save a model. 

ldim: dimension of the hidden layers of the Bayesian Neural Network (the likelihood model).

nlay: how many hidden layers.\*

epochs_used: Provide the number of epochs you want to run separately, for example 10 1000 . Please see lr option to understand.

lr_used: Provide the learning rate you want to use separately, for example  0.01 0.0001 -> epochs and lr are nested. This means we run optimization over 10 epochs with lr 0.01 and over 1000 epochs with lr 0.0001. (Note that with this structure you can perform step lr anneal).

anneal_used: Either to perform linear annealing on the last lr (in this case 0.0001) or not, as example Linear

batch_used: Batch size, as example 100

gpu_id: gpu id to use. Just provide the same number as nvidia_smi assigns

folder_name_provided: name of the folder to save log and experiments, for example 30MC_500epochs. Anyway the main file creates subfolders to further separate the models depending on the DNN model or the database_used. Thus, this folder can be created with specific information on how you estimate the ELBO.


\*We did a set of pilot studies with SVHN ADIENCE and CIFAR10. The code used only supports 2-hidden layers. In fact we observed that all the models could be calibrated with this topologies, only varying the number of neurons. Thus we performed the rest of experiments with this code. We then realized that for CIFAR100 a refactorization was needed to improve performance. However, using this code alter the way in which the random numbers were generated and thus implied redoing the experiments for CIFAR10, SVHN and ADIENCE. To avoid that, we just provide the two versions of code. This unefficient code was done as it was the best way to ensure we where doing things correctly, and avoid bugs that could potentially affect the results of our idea, thus making us reject it. After checking that our idea was correct we improve the code (both for this and BNNLR). In fact the main_ELBO_optimized.py can be use on CIFAR10 SVHN or ADIENCE but could give you different results from the ones reported in our main work.


## Compute the Predictive Distribution

Again depending if you used the optimized version or not here you have to run the main_predictive_inference.py or the main_predictive_inference_optimized.py files.

```
python main_predictive_inference.py --model_net [model_net] --data_dir [data_dir] --dataset [choosed_dataset] --model_dir [path_to_trained_BNN] --valid_test [validate_or_test] --MC_samples [MC]
```

### Code Params

model_net: name of the model that computed the logits, as example, densenet-121.

data_dir: location of your downloaded data for instance /tmp/data/

choosed_dataset: which dataset you want to use, as example, cifar10.

path_to_trained_BNN: absolute path to where the model has been saved (the one created when optimizing the ELBO). 

validate_or_test: either to perform validation or to run the test. If validation is provided code uses validation set to search for the optimal M value to compute the approximation of the integral of the predictive distribution that get better ECE15. If test is given it runs the test to approximate the integral.

MC: number of monte carlo samples to approximate the predictive distribution in test mode and then evaluate ECE15, and number of maximum samples to search for the optimal ECE15 in validation mode. It evaluates the ECE15 obtained from predictive distribution for samples in set {1,...,MC}.

n_gpu: gpu id to use. The one provided by nvidia_smi


## Bash Utilities

We give some bash utilities that can be used to run the experiments. Note that these utilities basically run python scripts and are prepared to change the learning rate, number of MC samples, epochs, dkl scale factor. So maybe you have to modify them for instance if you want to save a model each n epochs, or the number monte carlo used to search for the optimal in validation.


launch_train_experiments.sh: used to launch a bunch of experiments on one dataset and different models.

launch_valid_experiments.sh: used to launch experiments on validation. Basically you specify some training parameters: epochs MC_samples topology databases and models and it will do the rest for you. Note that this files expect a specific name for the folders: $dir$t"/"$mc"MC\_"$ep"eps_DKLSF"$dklsf"/", and give a specific name to the file where the results are saved: $dirResult$t"-"$mc"MC\_"$ep"eps\_"$dklsf"DKLSF". The good point is that if you do not alterate these two files (train and valid) the other files will do the rest for you. (parserPredictive and launch_test)

parsePredictive.sh: This files parse the output from launch_valid_experiments.sh and prepares everything to run the test. You only specify the database and the models and it will run the test for all the files created in validation.

launch_test_experiment.sh: This files will take as input the files created by parsePredictive.sh run the test and output a csv file. 

## Table with parameters of the reported models

The next table shows the configurations used to train each of the variational distributions of the experiments reported in this work. In cifar100 each run of the algorithm must save the model every 100 epochs ( --save_after [after_nepochs] ). The models trained are then used by the validation script. All the models use linear anneal on the last epochs (in the script set --anneal to Linear). To run experiments on ADIENCE pass gender as argument for choosed_dataset. All these models are trained with KL scale factor set to 0.1 (--dkl_scale_factor in the parser) and without warm-up (dkl_after_epochs set to -1).

Basically save_after means that the model is saved after 100 epochs. This was done because these models were more expensive to train. Thus, we pick a specific experiment (cifar100 wide-resnet-28x10 trained during 2000 epochs with 100MC samples for the ELBO) and during the optimization process we saved the model each 100 epochs (epochs: 100,200,300,400....2000). We then evaluate each of these models and select the best in validation. 

In main_predictive_inference.py, when you specify the model_dir there is a slight difference if you use --save_model_every or you do not. Assuming that, when training the variatonal distribution, you choosed MYMODEl as the argument for --folder_name you have to specify the directory to the inference script in the following way. As example assuming you save the model each 50 epochs and you want to evaluate the model on the 750 saved model, in main_predictive_inference, --model_dir should be specified as MYMODEL/750/ . If you do not choose to save the model each 100 just put MYMODEL:

    python main_predictive_inference_optimized.py --model_dir pathtoMYMODEL/MYMODEL/750/  #you set --save_model_every 50 when training the variatonal distribution
    python main_predictive_inference_optimized.py --model_dir pathtoMYMODEL/MYMODEL/     # you did not set it


The next table show training hyperparameters. In this case M for test refers to the optimal M found in validation for that model (we provide it so you can avoid its search although it is not expensive). Model at Epoch refers to the epoch where the model was saved. For example if we see Model at Epoch 1700 and Epochs 10 2000. This means we have to set --save_model_every 100 (or 50 or any factor from 1700). The code will save a model each 100/50 etc epochs and you have to evaluate performance on the 1700: --model_dir pathtoMYMODEL/MYMODEL/1700/


| Database  | Model | BNN Topology | Monte Carlo Samples | Epochs | Learning Rates | M for test |
| ------------- | ------------- |  ------------- |  ------------- |  ------------- | ------------- | ------------- | 
| CIFAR10  | WideResNet-28x10 | 10-512-512-10  |   30  |  110 | 0.001 | 15 |
| CIFAR10  | DenseNet-121 | 10-512-512-10  |  30 | 10   500 |  0.001   0.0001 | 15 |
| CIFAR10  | DenseNet-169 | 10-512-512-10  |  200 | 10   500 |  0.001   0.0001 | 15 |
| CIFAR10  | DualPathNet-92 | 10-512-512-10  |  200 | 50 |  0.001 | 19 |
| CIFAR10  | ResNet-101 |  10-512-512-10  |  200 | 10   500 |  0.001   0.0001 | 21 |
| CIFAR10  | VGG-19 | 10-512-512-10  |  200 | 10   500 |  0.001   0.0001 | 17 |
| CIFAR10  | PreactResNet-18 | 10-512-512-10  |  100 | 50 |  0.001 | 19 | 
| CIFAR10  | PreactResNet-164 | 10-512-512-10 | 30  | 30 | 0.001 | 17 |
| CIFAR10  | ResNext-29_8x16 |  10-1000-1000-10 | 400 | 20 | 0.001 | 25 |
| CIFAR10  | WideResNet-40x10 | 10-512-512-10  |  30 | 10   500 |  0.001   0.0001 | 16 |
| SVHN  | WideResNet-40x10 | 10-128-128-10  |  100  | 110  | 0.001 | 28 | 
| SVHN  | DenseNet-121 | 10-512-512-10 | 30  | 50 | 0.001  | 60 |
| SVHN  | DenseNet-169 | 10-1000-1000-10 | 200  | 110 | 0.001 | 55 |
| SVHN  | ResNet-50 | 10-512-512-10 | 30  | 10 200 | 0.001 0.0001 | 18 |
| SVHN  | PreactResNet-164 | 10-512-512-10 | 400  | 110 | 0.001 | 46 | 
| SVHN  | PreactResNet-18 | 10-512-512-10 | 800  | 15 | 0.001 | 23 |
| SVHN  | WideResNet-16x8 |10-512-512-10|  200 | 30  | 0.001 | 15 |
| SVHN  | WideResNet-28x10 | 10-128-128-10 | 200 | 110 | 0.001 | 15 | 
| ADIENCE | VGG-19 |  2-25-25-2  | 30 | 10 100 | 0.001 0.0001 | 21 |
| ADIENCE | DenseNet-121 | 2-10-10-2 | 30 | 50 | 0.001 | 227 |

| Database  | Model | BNN Topology | Monte Carlo Samples | Epochs | Learning Rates | Model at Epoch | M for test
| ------------- | ------------- |  ------------- |  ------------- |  ------------- | ------------- | ------------- | ------------- |
| CIFAR100  | WideResNet-28x10 | 100-2000-2000-100  |  100  | 10 2000  | 0.001 0.0001 | 1700 | 15 |
| CIFAR100  | DenseNet-121 | 100-2000-2000-100  |  100  | 10 2000  | 0.001 0.0001 | 1100 | 20 |
| CIFAR100  | DenseNet-169 | 100-2000-2000-100  |  100  | 10 2000  | 0.001 0.0001 | 1400 | 15 |
| CIFAR100  | ResNet-101 |  100-6000-6000-100  | 10 | 10 2000 | 0.001 0.0001 | 900 | 15 |
| CIFAR100  | VGG-19 |  100-1200-1200-100  |  100  | 10 2000  | 0.001 0.0001 | 100 | 197 |
| CIFAR100  | PreactResNet-18 |   100-2000-2000-100  |  100  | 10 2000  | 0.001 0.0001 | 2000 | 56 |
| CIFAR100  | PreactResNet-164 | 100-4500-4500-100 | 30 | 10 3000 | 0.001 0.0001 | 2100 |53 |
| CIFAR100  | ResNext-29_8x16 | 100-5000-5000-100  | 20 | 10 2000 |0.001 0.0001 | 1900 | 15 |
| CIFAR100  | WideResNet-40x10 | 100-2000-2000-100  |  100  | 10 2000  | 0.001 0.0001 | 1600 | 49 |



