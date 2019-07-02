echo "This experiment is done with the non-optimized version of the code. It will not run as quicker as the other versions, BNN Local Rep for instance"

echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "--------------------------------------INSTALLING PYTORCH ----------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
virtualenv -p python2.7 /tmp/0.3.1_cuda8_pytorch
source /tmp/0.3.1_cuda8_pytorch/bin/activate
pip install  http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl

echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "---------------------DOWNLOADING DATA: EXPERIMENT ON CIFAR10 PREACTRESNET-164--------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"

wget "https://drive.google.com/uc?export=download&id=1j3wMT_WyCI-weyIEByUC_gDck6JF3tyb" -O cifar10_preactresnet-164_logit_prediction_train.npy
wget "https://drive.google.com/uc?export=download&id=1-Isl1Kqd8FOzYleebfp3IqF-bUlK95Po" -O cifar10_preactresnet-164_logit_prediction_test.npy
wget "https://drive.google.com/uc?export=download&id=1prdV7bR9TERgYOnxA4yDYTBRS1HzAs3H" -O cifar10_preactresnet-164_logit_prediction_valid.npy
wget "https://drive.google.com/uc?export=download&id=1581eW3Lo4KbzZIyF_ZvkWwDUKb3oqx0v" -O cifar10_preactresnet-164_true_test.npy
wget "https://drive.google.com/uc?export=download&id=1NI0uP46ApR_ZQtEmiew0kMB_r8daD7wN" -O cifar10_preactresnet-164_true_train.npy
wget "https://drive.google.com/uc?export=download&id=1z5Fbt3bgWhMCYeukvCYeSPAHt3GDh_k-" -O cifar10_preactresnet-164_true_valid.npy

echo "----------------------------FINISH DOWNLOADING---------------------------------------------------------"


mkdir -p "./data/preactresnet-164_cifar10/"
mv *.npy ./data/preactresnet-164_cifar10/

echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "----------------------------RUNNING TEMP SCALING ------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"

python ../main_plat_scaling.py --model_net preactresnet-164 --data_dir ./data/ --dataset cifar10 --epochs 500 | tee aux_plat_scaling_file

echo "----------------------------FINISH TEMP SCALING--------------------------------------------------------"

accuracy_plat=`cat aux_plat_scaling_file | grep "Getting accuracy" | cut -d" " -f4 | awk 'BEGIN{count=0}{if (count==3){test_error=$1};count+=1}END{print test_error}'`

cat aux_plat_scaling_file | grep "Getting and ECE" | cut -d" " -f13 | awk -v acc_plat="$accuracy_plat" 'BEGIN{print "=======TEMP SCAL=====\n test  uncalibrated|||test  calibrated||| accuracy test \n";count=0;aux=""}{if (count>1){aux=aux"     "$1};count+=1}END{print aux"    "acc_plat"\n\n"}' > print_results


echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "----------------------------RUNNING BNN ---------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"


./launch_train_experiments.sh

echo "----------------------------FINISH TRAINING THE VARiATIONAL DISTRIBUTION-------------------------------"


echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "----------------------------RUNNING PREDICTIVE DISTRIBUTION--------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"

./launch_valid_predictive.sh

./parsePredictive.sh

./launch_test_predictive.sh

echo "----------------------------RUNNING PREDICTIVE DISTRIBUTION-------------------------------"

uncalibrated=`cat print_results | awk 'BEGIN{count=0}{if (count==3){print $0};count+=1}' | sed 's/    */ /g'| cut -d " " -f2`


cat TestPredictivecifar10preactresnet-164.txt | sed 's/,/ /g' | awk -v uncalibrated="$uncalibrated" 'BEGIN{count=0; print "=======  BNN  =====\n test  uncalibrated|||test  calibrated||| accuracy test\n"}{if (count==1){print "     "uncalibrated"     "$6"    "$7};count+=1}'  >> print_results


cat print_results


rm print_results
rm ParsePredictivecifar10preactresnet-164.txt
rm TestPredictivecifar10preactresnet-164.txt
rm -r predictiveResult
rm -r pretrain_models
rm aux_plat_scaling_file
rm -r data/

