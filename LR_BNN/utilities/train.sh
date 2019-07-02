dataset='putdatasethere'
model_list=( 'deepmodel1' 'deepmodel2' 'deepmodel3')
gpunumber=0
for model in "${model_list[@]}"
do

./launch_train_experiment.sh $dataset $model $gpunumber

done


