task=$1
model=$2
model_arch=$3

if [[ ${model_arch} == 'opt' ]]; then
    input_data=data/${task}_opt.jsonl
else
    input_data=data/${task}.jsonl
fi
export CUDA_VISIBLE_DEVICES=4
python3 -u run_helm.py \
    --input_path ${input_data} \
    --output_path ${task}-${model_arch}-h2o.jsonl \
    --model_name ${model} \
    --model_arch ${model_arch} \
    --enable_small_cache \
    --heavy_ratio 0.2 \
    --recent_ratio 0.1 \
    --initial_ratio 0.1

