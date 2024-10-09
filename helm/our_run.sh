jsonl=../xsum-llama-h2o.jsonl # 我们假设此前生成的结果文件在 h2o_hf 下，你可以更改此路径
task=xsum
model_arch=llama
output_name=xsum-llama_h2o0.2-test # 仅控制评估指标的输出路径，可自行更改

# 将此前生成的结果文件与 任务数据集的request 打包，整理为适当格式
python scripts/offline_eval/import_results.py together ${jsonl} --cache-dir prod_env/cache 

# 执行第三方库的 helm-run 指令进行评估
helm-run --conf src/helm/benchmark/presentation/${task}/run_specs_${model_arch}.conf --local --max-eval-instances 1000 --num-train-trials=1 --suite ${output_name} -n 1

# 执行 helm-summarize 指令汇总结果
helm-summarize --suite ${output_name}