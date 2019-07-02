
dataset=$1
model=$2
n_gpu=$3
#training_schemes=( Montecarlo_samples neurons layers training_schemes dkl_scale_factor)
training_schemes=(
#1MC 32 neurons 1 layer
"100 2048 2 10_1000 0.01_0.001 0.1" 
 )

for ts in "${training_schemes[@]}"
do


	MC=`echo $ts | cut -d" " -f1`
        neurons=`echo $ts | cut -d" " -f2`
        layers=`echo $ts | cut -d" " -f3`
        epochs=`echo "$ts" | cut -d" " -f4 | awk 'BEGIN{p_var=""}{split($0,a,"_");for( i in a){p_var=p_var" "a[i]}}END{print p_var}'`
        lr=`echo "$ts" | cut -d" " -f5 | awk 'BEGIN{p_var=""}{split($0,a,"_");for( i in a){p_var=p_var" "a[i]}}END{print p_var}'`
	dklsf=`echo $ts | cut -d" " -f6`
	total_epochs=`echo $epochs | awk 'BEGIN{total=0}{for( i=1; i<= NF; i++){total+=$i}}END{print total}'`

        python ../main.py  --model_net $model --data_dir ../data/ --dataset $dataset --MC_samples $MC  --layer_dim $neurons --n_layers $layers  --epochs $epochs --lr $lr --dkl_scale_factor $dklsf  --batch 100 --anneal Linear --n_gpu $n_gpu --folder_name  $MC"MC_"$total_epochs"eps_DKLSF"$dklsf 


done


