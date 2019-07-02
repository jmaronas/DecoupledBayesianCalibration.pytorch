
database='cifar10'
model='wide-resnet-40x10'
inputdim='10'


dir="./pretrain_models/"$database"/"$model"/"
dirResult="./predictiveResult/"$database"/"$model"/"

mkdir -p $dirResult

MC_samples=( 1 )
epochs=( 10  )
topology=( 
"${inputdim}_128_${inputdim} 1 128 0.01"
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

       python  ../main_predictive_inference_withLR.py --model_net $model --data_dir ./data/ --dataset $database --n_gpu 0 --valid_test valid --MCsamples 3000 --n_layers $l --layer_dim $n --model_dir $dir$t"/"$mc"MC_"$ep"eps_DKLSF"$dklsf"/models/BNN.pth.tar" | tee $dirResult$t"-"$mc"MC_"$ep"eps_"$dklsf"DKLSF"

   done
   done
done



