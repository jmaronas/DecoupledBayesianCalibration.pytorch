echo "This experiment is done with the non-optimized version of the code. It will not run as quicker as the other versions, BNN Local Rep for instance"

echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "--------------------------------------INSTALLING PYTORCH ----------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
virtualenv -p python2.7 /tmp/0.4.0_cuda9.1_pytorch
source /tmp/0.4.0_cuda9.1_pytorch/bin/activate
pip install  https://download.pytorch.org/whl/cu91/torch-0.4.0-cp27-cp27mu-linux_x86_64.whl
pip install numpy
deactivate

virtualenv -p python2.7 /tmp/0.3.1_cuda8_pytorch
source /tmp/0.3.1_cuda8_pytorch/bin/activate
pip install  http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl

echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "---------------------DOWNLOADING DATA: EXPERIMENT ON CIFAR10 WIDE-RESNET-40x10--------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"

wget "https://drive.google.com/uc?export=download&id=180qO-m_FUMixVb7AIDe7RQnhCsVJQSoZ" -O cifar10_wide-resnet-40x10_logit_prediction_train.npy
wget "https://drive.google.com/uc?export=download&id=1q6zPHffw7O3k1SJvtz_rDLzYkw4WeZz0" -O cifar10_wide-resnet-40x10_logit_prediction_test.npy
wget "https://drive.google.com/uc?export=download&id=1LAgKyJZSfwkTFLOhyu-VxLyS7XmWXbMi" -O cifar10_wide-resnet-40x10_logit_prediction_valid.npy
wget "https://drive.google.com/uc?export=download&id=1H_NoPC5CyeQJyEiJRxdUmccOSPF8W8ls" -O cifar10_wide-resnet-40x10_true_test.npy
wget "https://drive.google.com/uc?export=download&id=1V4AUlCjz2IlRzRiC--8SHaxA26y6xS2f" -O cifar10_wide-resnet-40x10_true_train.npy
wget "https://drive.google.com/uc?export=download&id=13yuza28BAQKqwyhcZysXSGbBAz1zuAdj" -O cifar10_wide-resnet-40x10_true_valid.npy

echo "----------------------------FINISH DOWNLOADING---------------------------------------------------------"


mkdir -p "./data/wide-resnet-40x10_cifar10/"
mv *.npy ./data/wide-resnet-40x10_cifar10/

echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "----------------------------RUNNING TEMP SCALING ------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"

python ../../standard_BNN/main_plat_scaling.py --model_net wide-resnet-40x10 --data_dir ./data/ --dataset cifar10 --epochs 200 | tee aux_plat_scaling_file

deactivate
source /tmp/0.4.0_cuda9.1_pytorch/bin/activate

echo "----------------------------FINISH TEMP SCALING--------------------------------------------------------"

accuracy_plat=`cat aux_plat_scaling_file | grep "Getting accuracy" | cut -d" " -f4 | awk 'BEGIN{count=0}{if (count==3){test_error=$1};count+=1}END{print test_error}'`

cat aux_plat_scaling_file | grep "Getting and ECE" | cut -d" " -f13 | awk -v acc_plat="$accuracy_plat" 'BEGIN{print "=======TEMP SCAL=====\n test  uncalibrated|||test  calibrated||| accuracy test \n";count=0;aux=""}{if (count>1){aux=aux"     "$1};count+=1}END{print aux"    "acc_plat"\n\n"}' > print_results


echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "----------------------------RUNNING BNN ---------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"
echo "-------------------------------------------------------------------------------------------------------"


./train.sh

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


cat TestPredictivecifar10wide-resnet-40x10.txt | sed 's/,/ /g' | awk -v uncalibrated="$uncalibrated" 'BEGIN{count=0; print "=======  BNN  =====\n test  uncalibrated|||test  calibrated||| accuracy test\n"}{if (count==1){print "     "uncalibrated"     "$6"    "$7};count+=1}'  >> print_results


cat print_results

rm print_results
rm ParsePredictivecifar10wide-resnet-40x10.txt
rm TestPredictivecifar10wide-resnet-40x10.txt
rm -r predictiveResult
rm -r pretrain_models
rm aux_plat_scaling_file
rm -r data/

