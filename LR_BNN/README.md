# BNN Local Reparameterization (BNNLR)
 
We provide the basic information to reproduce the results for BNNLR. Refer to the main (anonymized for the moment) work to check which models you can run under this directory . Execute python file.py --help for further information (for example on which kind of information or possibilities can be passed to the python parser)

As T Scal provides a unique optimum given a set of data just go to the standard_BNN folder to get the code an explanation for running T scal.

## Material provided

Code for running experiments for BNNLR

Some bash utilities

A folder with a particular example that does everything for you.

This code provides some utilities that standard_BNN does not. Check below. These  utilities only facilitate tasks involving lots of experiments.

## Software required

This code is prepared to be used with pytorch version 0.4.0. Activate your virtual enviroment before running any of the files provided.

Use bash instead of dash. This is not a limitation but is preferable if you are going to use the bash scripts provided.

## Replicate an example of the paper.

We provide an easy way to replicate one of the experiment of the paper. To do this you do not have to install anything, we do it for you. Just go inside the example folder and execute example.sh. This file will download pytorch, download a set of data, train temperature scaling and train a BNN applying LR. You can follow the guidelines from this example to replicate any of the experiments provided.

Guidelines (assuming you meet the hardware requirements and use linux):
 
 ```
 git clone https://github.com/https://github.com/jmaronas/DecoupledBayesianCalibration.pytorch.git
 cd ./https://github.com/jmaronas/DecoupledBayesianCalibration.pytorch.git/LR_BNN/example
 ./example.sh

 ```

## Baseline Results

Follow the steps in the standard_BNN folder

## Training Variational Distribution (optimize the ELBO aplying Local Reparameterization)

```
python main_ELBO_withLR.py  --model_net [model_net] --data_dir [data_dir] --dataset [choosed_dataset] --MC_samples [MC] --layer_dim [ldim] --n_layers [nlay]  --save_after [after_nepochs] --save_model_every [each_nepochs] --epochs [epochs_used] --lr [lr_used] --batch  [batch_used] --anneal [anneal_used] --n_gpu [gpu_id] --folder_name [folder_name_provided] --dkl_after_epoch [DAE] --dkl_scale_factor [DSF] --prior_is_learnable [islearnable] 
```

### Code Params

model_net: name of the model that computed the logits, as example, densenet-121

data_dir: location of your downloaded data, as example, /tmp/data/

choosed_dataset: which dataset you want to use, as example, cifar10

MC: number of montecarlo samples to estimate the likelihood expectation under the variational distribution, as example, 30.

ldim: dimension of the hidden layers of the Bayesian Neural Network (the likelihood model).

nlay: how many hidden layers.

after_nepochs:  save the model after_nepochs. It is disabled if you set a value for --save_model_every. 

each_nepochs: save model each n epochs. We explain the usage of these two arguments in the experiment section.

epochs_used: Provide the number of epochs you want to run separately, for example 10 1000 . Please see lr option to understand.

lr_used: Provide the learning rate you want to use as a list, for example  0.01 0.0001 -> epochs and lr are nested. This means we run optimization over 10 epochs with lr 0.01 and over 1000 epochs with lr 0.0001. (Note that with this structure you can perform step lr anneal).

anneal_used: Either to perform linear annealing on the last lr (in this case 0.0001) or not, as example Linear

batch_used: Batch size, as example 100

gpu_id: gpu id to use. Just provide the same number as nvidia_smi assigns


folder_name_provided: name of the folder to save log and experiments, for example 30MC_500epochs. Anyway the main file creates subfolders to further separate the models depending on the DNN model or the database_used. Thus, this folder can be created with specific information on how you estimate the ELBO.

DAE: after which epoch we also optimize the DKL (known as warm up). If provide -1 it add the DKL term from the beginning.

DSF: factor to scale the DKL (\beta in the paper), as example 0.1

islearnable: if provided this argument makes the parameters of the prior distribution learnables.

## Compute the Predictive Distribution

This code runs an approximation to equation 1 in the paper. Monte Carlo integration.

```
python main_predictive_inference_withLR.py --model_net [model_net] --data_dir [data_dir] --dataset [choosed_dataset] --model_dir [path_to_trained_BNN] --valid_test [validate_or_test] --MCsamples [MC]  --layer_dim [ldim] --n_layers [nlay]  --prior_is_learnable [islearnable] 

```

### Code Params

model_net: name of the model that computed the logits, as example, densenet-121.

data_dir: location of your downloaded data.

choosed_dataset: which dataset you want to use, as example, cifar10.

path_to_trained_BNN: absolute path to where the model has been saved (the one created when optimizing the ELBO --folder-name_provided)

validate_or_test: either to perform validation or to run the test. If validation is provided code uses validation set to search for the optimal M value to compute the approximation of the integral of the predictive distribution that get better ECE15. If test is given it runs the test to approximate the integral.

MC: number of monte carlo samples to approximate the predictive distribution in test mode and then evaluate ECE15, and number of maximum samples to search for the optimal ECE15 in validation mode. It evaluates the ECE15 obtained from predictive distribution for samples in set {1,...,MC}.

n_gpu: gpu id to use. The one provided by nvidia_smi

ldim: dimension of the hidden layers of the Bayesian Neural Network (the likelihood model).

nlay: how many hidden layers.

islearnable: this argument must be provided if you optimize the prior distribution in the learned variatonal model.

## Bash Utilities

We give some bash utilities that can be used to run the experiments. Note that these utilities basically run the python and are prepared to change the learning rate, number of MC samples, epochs, dkl scale factor. So maybe you have to modify them for instance if you want to change the number monte carlo used to search for the optimal in validation.

check_experiment_finished.py: to check for nan files or experiments that did not finish, for instance if the machine where you are running the model shuts down.

launch_train_experiments.sh: used to launch a bunch of experiments on one dataset and different models.

train.sh: to launch launch_train_experiments.sh over different datasets or models

launch_valid_experiments.sh: used to launch experiments on validation. Basically you specify some training parameters: epochs MC_samples topology databases and models and it will do the rest for you. Note that this files expect a specific name for the folders: $dir$t"/"$mc"MC\_"$ep"eps_DKLSF"$dklsf"/", and give a specific name to the file where the results are saved: $dirResult$t"-"$mc"MC\_"$ep"eps\_"$dklsf"DKLSF". The good point is that if you do not alterate these two files (train and valid) the other files will do the rest for you.

launch_valid_experiments_save_per_epochs: same as above but for model saved after n_epochs

parsePredictive.sh: This files parse the output from launch_valid_experiments.sh and prepares everything to run the test. You only specify the database and the models and it will run the test for all the files created in validation.

parsePredictive_save_per_epochs.sh:  same as above but for model saved after n_epochs

launch_test_experiment.sh: This files will take as input the files created by parsePredictive.sh run the test and output a csv file. 

launch_test_experiment_save_per_epochs.sh: same as above but for model saved after n_epochs


## Table with parameters of the reported models

The next table shows the configurations used to train each of the variational distributions of the experiments reported in this work. The models trained are then used by the validation script. All the models use linear anneal on the last epochs (in the script set --anneal to Linear). To run experiments on ADIENCE pass gender as argument for the database. All these models are trained with  without warm-up (dkl_after_epochs set to -1). The rest of parameters are provided in the table.

There is a slight difference with the code from standard_BNN. In this case when evaluating the model you have to pass the absolute path to the .pth.tar file (the file that contains the saved parameters) in the --model_dir [path_to_trained_BNN] argument. For instance if --folder_name [folder_name_provided] in the training script was MYMODEL you will have to pass: ABSOLUTEPATH/MYMODEL/models/BNN.pth.tar. to the inference script. This is independent if you have saved the model each n epochs or not. (--save_model_every).

As we mentioned above there is a difference when using --save_model_every or --save_after. The latter is used just as a checkpoint, so you can restart the training process if your computer shuts down. The earlier is used to save the model each n epochs (we provide the explanation in the standard_BNN readme). In this case, as example, remember you must provide the absolute path.

    python main_predictive_inference_optimized.py --model_dir pathtoMYMODEL/MYMODEL/750/BNN_epoch750.pth.tar  #you set --save_model_every 50 when training the variatonal distribution
    python main_predictive_inference_optimized.py --model_dir pathtoMYMODEL/MYMODEL/models/BNN.pth.tar     # you did not set it


In cifar100 each run of the algorithm must save the model every 50 epochs ( --save_after [after_nepochs] ).

| Database  | Model | BNN Topology | Monte Carlo Samples | Epochs | Learning Rates | KL SCALE FACTOR | M for test | Model at Epoch | 
| ------------- | ------------- |  ------------- |  ------------- |  ------------- | -------------  | ------------- | ------------- | ------------- | 
| CIFAR10  | WideResNet-28x10 | 10-512-10 | 1 | 50 | 0.01 | 0.01  | 15 | - |
| CIFAR10  | DenseNet-121 | 10-32-32-10 | 1 | 50 | 0.01 | 0.1 |  23 | - |
| CIFAR10  | DenseNet-169 | 10-64-64-10 | 1 | 10 500 | 0.01  0.001 | 0.1 | 17 | - |
| CIFAR10  | DualPathNet-92 | 10-512-512-10 | 1 | 10 1000 | 0.01 0.001 | 0.1 |  20 | - | 
| CIFAR10  | ResNet-101 |  10-64-64-10 | 10 | 10 1000 | 0.01 0.001 | 0.1 | 16 | 950 |
| CIFAR10  | VGG-19 | 10-64-64-10 | 30 | 10 100 | 0.01 0.001 | 0.1 |  15 | 95 |
| CIFAR10  | PreactResNet-18 |  10-128-128-10 | 1 | 10 100 | 0.01 0.001 | 0.1 |  20 | - |
| CIFAR10  | PreactResNet-164 | 10-24-24-10 | 30 | 10 100 | 0.01 0.001 | 0.1 |  65 | - |
| CIFAR10  | ResNext-29_8x16 |  10-128-10 | 30 | 10 100 | 0.01 0.001 | 0.01 |  17 | - |
| CIFAR10  | WideResNet-40x10 | 10-128-10 | 1 | 10 | 0.01 | 0.01 | 78 | - |
| SVHN  | WideResNet-40x10 |  10-64-64-10 | 10 | 10 100 | 0.01 0.001 | 0.1 |  16 | - |
| SVHN  | DenseNet-121 | 10-64-64-10 | 30 | 10 100 | 0.01 0.001 | 0.1 | 22 | - |
| SVHN  | DenseNet-169 | 10-64-64-10 | 10 | 10 | 0.01 | 0.1 |  79 | - |
| SVHN  | ResNet-50 | 10-128-10 | 10 | 10 500 | 0.01 0.001 | 0.01 |  18 | - |
| SVHN  | PreactResNet-164 |  10-32-32-10 | 1 | 10 1000 | 0.01 0.001 | 0.1 | 55 | - |
| SVHN  | WideResNet-16x8 |  10-128-10 | 100 | 10 1000 | 0.01 0.001 | 0.01 | 49 | 800 |
| SVHN  | PreactResNet-18 | 10-48-48-10 | 10 | 10 500 | 0.01 0.001 | 0.1 |  33 | - |
| SVHN  | WideResNet-28x10 |  10-64-64-10 | 30 | 10 500 | 0.01 0.001 | 0.1 | 32 | - | 
| ADIENCE | VGG-19 | 2-64-64-2 | 10 | 10 1000 | 0.01 0.001 | 1 | 68 | - |
| ADIENCE | DenseNet-121 | 2-64-64-2 | 10 | 10 500 | 0.01 0.001 | 1 |  32 | - |
| VGGFACE2  | MobileNet |  2-32-2 | 30 | 10 500 | 0.01 0.001 | 0.1 | 24 | - |
| VGGFACE2  | SeNet | 2-64-64-2 | 1 | 10 100 | 0.01 0.001 | 1 | 47 | - |
| VGGFACE2  | VGG | 2-32-32-2 | 1 | 10 500 | 0.01 0.001 | 0.1 |  35 | - |
| CIFAR100  | WideResNet-28x10 | 100-2048-100 | 50 | 10 1000 | 0.01 0.001 | 0.01 |  15 | 250 |
| CIFAR100  | DenseNet-121 | 100-128-100 | 50 | 10 1000 | 0.01 0.001 | 0.01 |  16 | 750 |
| CIFAR100  | DenseNet-169 | 100-512-100 | 50 | 10 1000 | 0.01 0.001 | 0.01 |  17 | 750 |
| CIFAR100  | ResNet-101 | 100-128-100 | 200 | 10 2000 | 0.01 0.001 | 0.01 |  1874 | 750 |
| CIFAR100  | VGG-19 | 100-25-100 | 1 | 10 1000 | 0.01 0.001 | 0.01 | 614 | 50 |
| CIFAR100  | PreactResNet-18 |  100-256-100 | 100 | 10 1000 | 0.01 0.001 | 0.01 |  84 | 200 |
| CIFAR100  | PreactResNet-164 | 100-1024-100 | 50 | 10 1000 | 0.01 0.001 | 0.01 |  561 | 450 |
| CIFAR100  | ResNext-29_8x16 | 100-5000-100 | 50 | 10 1000 | 0.01 0.001 | 0.01 | 6 | 750 | 
| CIFAR100  | WideResNet-40x10 | 100-512-100 | 100 | 10 1000 | 0.01 0.001 | 0.01 |  48 | 850 |
| CARS  | DenseNet-169\* |  196-4096-4096-196 | 50 | 50 | 0.01 | 0.01 |  44 | 40 |
| CARS  | DenseNet-121 |  196-196 | 200 | 10 100 | 0.1 0.01 | 0.0001 |  881 | 10 |
| CARS  | ResNet-18 | 196-196 | 200 | 10 15 | 0.1 0.01 | 0.0001 | 407 | 20 |
| CARS  | ResNet-50 | 196-196 | 200 | 10 15 | 0.1 0.01 |  0.0001 |  456 | 15 |
| CARS  | ResNet-101\* | 196-4096-4096-196 | 50 | 50 | 0.01 | 0.1 |  376 | 10 |
| BIRDS  | DenseNet-169 | 200-200 | 400 | 10 100 | 0.1 0.01 | 0.0001 | 2606 | 20 |
| BIRDS  | DenseNet-121 | 200-200 | 100 | 10 100 | 0.1 0.01 | 0.0001 | 957 | 45 |
| BIRDS  | ResNet-18 | 200-200  | 100 | 10 2000 | 0.1 0.01 | 0.0001 | 58 | 1950 |
| BIRDS  | ResNet-50 | 200-200  | 200  | 10 1000 | 0.1 0.01 | 0.0001 |  273 | 250 |
| BIRDS  | ResNet-101 | 200-200 | 100 | 10 100 | 0.1 0.01 | 0.0001 | 40 | 90 |

\*Prior is learned



