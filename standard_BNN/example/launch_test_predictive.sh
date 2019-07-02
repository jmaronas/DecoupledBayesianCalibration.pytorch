database='cifar10'
modelall=('preactresnet-164')
inputdim='10'

for model in ${modelall[@]}
do
dir="./pretrain_models/"$database"/"$model"/"


#only for small files if not parse by line
file_head=`cat "./ParsePredictive"$database$model".txt"  | awk 'BEGIN{}{print $0;exit}'`
file=`cat "./ParsePredictive"$database$model".txt"  | awk 'BEGIN{}{if (NR>1){print $0}}'`

echo $file_head > "TestPredictive"$database$model".txt" 
IFS=','
while read neuron layers mcsamplestrain epochs dklsf  null1 null2  ecevalid mcvalid ; do

if [ "$layers" -eq 2 ]; then
top=$neuron"_"$neuron
elif [ "$layers" -eq 3 ]; then
top=$neuron"_"$neuron"_"$neuron
else
top=$neuron
fi

model_dir=$dir$inputdim"_"$top"_"$inputdim

var=`python ../main_predictive_inference.py --model_net $model --data_dir ./data/ --dataset $database --valid_test test --MC_samples $mcvalid --model_dir $model_dir"/"$mcsamplestrain"MC_"$epochs"eps_DKLSF"$dklsf"/" |  awk 'BEGIN{}{print $11","$15","substr($18,1,length($18)-1)}'`

read ece _ acc <<< "$var"


echo $neuron","$layers","$mcsamplestrain","$epochs","$dklsf","$ece","$acc","$ecevalid","$mcvalid >> "TestPredictive"$database$model".txt"


done  <<< "$file"
done
