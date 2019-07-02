
database='putdatasethere'
model='putdatasethere'
inputdim='putdimhere'


dir="./pretrain_models/"$database"/"$model"/"
dirResult="./predictiveResult/"$database"/"$model"/"

mkdir -p $dirResult

MC_samples=( 1 10 30 )
epochs=( 10 50 110 510 1010 )
topology=(
"${inputdim}_32_${inputdim} 1 32 1" 

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

       python ../predictive_estimation.py --model_net $model --data_dir ../data/ --dataset $database --n_gpu 0 --valid_test valid --MCsamples 3000 --n_layers $l --layer_dim $n --model_dir $dir$t"/"$mc"MC_"$ep"eps_DKLSF"$dklsf"/models/BNN.pth.tar" | tee $dirResult$t"-"$mc"MC_"$ep"eps_"$dklsf"DKLSF"

   done
   done
done



