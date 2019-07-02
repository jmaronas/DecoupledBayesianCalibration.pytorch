#####################CIFAR10
dataset='cifar10'
model_list=( 'wide-resnet-40x10')
gpunumber=0
for model in "${model_list[@]}"
do

./launch_train_experiment.sh $dataset $model $gpunumber

done

