
database='putdatasethere'
modelall=( 'deepmodel1' 'deepmodel2')
inputdim='dimension'

for model in "${modelall[@]}"
do

dir="./pretrain_models/"$database"/"$model"/"
dirResult="./predictiveResult/"$database"/"$model"/"

mkdir -p $dirResult

MC_samples=(30)
epochs=( 50 110 510 1010)
topology=(
"${inputdim}_5_5_${inputdim} 2 5 0.1"

#this will evaluate the models trained with 30 MC with the specified epochs and topologies with two hidden layers of 5 neurons, with DKLscale factor=0.1
)



for mc in "${MC_samples[@]}"
do
   for ep in "${epochs[@]}"
   do
     for top in "${topology[@]}"
     do

	t=`echo $top | cut -d " " -f 1`
        l=`echo $top | cut -d " " -f 2`
        n=`echo $top | cut -d " " -f 3`
        dklsf=`echo $top | cut -d " " -f 4`

       python main_predictive_inference.py --model_net $model --data_dir ./data/ --dataset $database  --valid_test validation --MC_samples 3000 --model_dir $dir$t"/"$mc"MC_"$ep"eps_DKLSF"$dklsf"/" | tee $dirResult$t"-"$mc"MC_"$ep"eps_"$dklsf"DKLSF"


   done
   done
done

done
