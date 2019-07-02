database='putdatasethere'
model='putmodelhere'

dirPredictive="./predictiveResult/"$database"/"$model"/"

echo "Neurons per layer,layers,MonteCarlo,Epochs,DKLSF,ECE15,Accuracy,ECE15Valid,MCValid" >   "ParsePredictive"$database$model".txt"

for file in `ls $dirPredictive`
do

topology=`echo $file | cut -d"-" -f1`
number_layers=`echo $topology | grep -o "_" | wc -l`
number_layers="$(($number_layers -1))"
neurons_per_layer=`echo $topology | cut -d"_" -f2`

algo_params=`echo $file | cut -d"-" -f2`
mc_samples=`echo $algo_params | cut -d "_" -f1 | cut -d "M" -f1 `
epochs=`echo $algo_params | cut -d "_" -f2 | cut -d "e" -f1 `
dklsf=`echo $algo_params | cut -d "_" -f3 | cut -d "D" -f1 `


cat $dirPredictive"/"$file  | grep "Best" | awk -v dklsf="$dklsf" -v var="$file" -v mc_samples="$mc_samples" -v epochs="$epochs" -v num_layers="$number_layers" -v neurons_per_layer="$neurons_per_layer" 'BEGIN{}{print neurons_per_layer","num_layers","mc_samples","epochs","dklsf", , ,"$7","$10} ' >> "ParsePredictive"$database$model".txt"


done

