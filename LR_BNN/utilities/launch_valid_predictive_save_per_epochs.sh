database='putdatasethere'
model='putmodelhere'
inputdim='putdimhere'

dir="./pretrain_models/"$database"/"$model"/"
dirResult="./predictiveResult/"$database"/"$model"/"

mkdir -p $dirResult

MC_samples=(100)
epochs=( 1010 )
topology=(
"${inputdim}_128_${inputdim} 1 128 1"
)
save_after=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000)



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

        for ep_saved in "${save_after[@]}"
          do

            python ../predictive_estimation.py --model_net $model --data_dir ../data/ --dataset $database --n_gpu 0 --valid_test valid --MCsamples 3000 --n_layers $l --layer_dim $n --model_dir $dir$t"/"$mc"MC_"$ep"eps_DKLSF"$dklsf"/"$ep_saved"/models/BNN_epoch"$ep_saved".pth.tar" | tee $dirResult$t"-"$mc"MC_"$ep"eps_"$dklsf"DKLSF_"$ep_saved"saved.after"


         done
     done
   done
done



