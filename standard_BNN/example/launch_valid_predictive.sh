
database='cifar10'
modelall=( 'preactresnet-164')
inputdim='10'

for model in "${modelall[@]}"
do

dir="./pretrain_models/"$database"/"$model"/"
dirResult="./predictiveResult/"$database"/"$model"/"

mkdir -p $dirResult

MC_samples=(30)
epochs=( 30 )
topology=(
"${inputdim}_512_512_${inputdim} 2 512 0.1"
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

       python ../main_predictive_inference.py --model_net $model --data_dir ./data/ --dataset $database  --valid_test validation --MC_samples 3000 --model_dir $dir$t"/"$mc"MC_"$ep"eps_DKLSF"$dklsf"/" | tee $dirResult$t"-"$mc"MC_"$ep"eps_"$dklsf"DKLSF"


   done
   done
done

done
