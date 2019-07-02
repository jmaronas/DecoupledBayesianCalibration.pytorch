dataset='putdatasethere'
modelall=('deepmodel1' 'deepmodel2')
n_gpu=0

for model in "${modelall[@]}"
do
#training_schemes=( Montecarlo_samples neurons layers training_parameters DKL scale factor)
training_schemes=(
#"30 5 2 10_1000 0.001_0.0001 0.1" Separate with _ if you want to perform step learning rate anneal
"30 512 2 110 0.001 0.1" 
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


	python main_ELBO.py --model_net $model --n_layers $layers --layer_dim $neurons --dkl_scale_factor $dklsf --dkl_after_epoch -1 --batch 100 --lr $lr --epochs $epochs --MC_samples $MC --n_gpu $n_gpu --anneal Linear --data_dir PUT_DATA_DIR_HERE --dataset $dataset --folder_name $MC"MC_"$total_epochs"eps_DKLSF"$dklsf

done

done

