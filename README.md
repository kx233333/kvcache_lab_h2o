# KVCache LAB 实验配置指南

本lab基于论文 *H2O: Heavy-Hitter Oracle for Efficient Generative
Inference of Large Language Models*，采用论文中的helm评估框架，模型使用`llama-7b`。

本readme介绍如何配置H2O论文代码的实验运行环境，并用helm框架对生成结果进行评估。

(已在 .48 服务器单卡4090，cuda 12.1，python 3.8 环境下完成复现)


## 1 配置实验环境

- 新建conda环境 (crfm-helm==0.2.3 需要 python~=3.8)

  > conda creat -n \<your name> python=3.8

- 打开环境

  > conda activate \<your name>

- 安装 pytorch

  > conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

- 安装依赖库

  ```
  pip install crfm-helm==0.2.3
  pip install transformers==4.33
  ```

  - （注1：需要先安装crfm-helm。安装transformers 时会出现以下报错，可以直接忽略）

    > ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.  
    > crfm-helm 0.2.3 requires transformers~=4.28.1, but you have transformers 4.33.0 which is incompatible.  

  

  - （注2：安装 crfm-helm 会把 torchvision 等回滚到旧版本，导致此环境下能够使用 helm-run 等评估指令对已生成的result进行评估，但无法运行run_helm.py。为此，需要重新执行 

    > conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia 

    以再次安装pytorch。最初几次可能会汇报

    > Solving environment: done  
    > \# All requested packages already installed.

    忽略此提示，尝试`conda list`检查环境，再反复执行命令直到出现正常的安装提示，如下：

    ```
    The following NEW packages will be INSTALLED:
    
      pytorch            pytorch/linux-64::pytorch-2.4.0-py3.8_cuda12.1_cudnn9.1.0_0 
      sympy              pkgs/main/linux-64::sympy-1.12-py38h06a4308_0 
      torchvision        pytorch/linux-64::torchvision-0.19.0-py38_cu121 
    ```

    输入y确认，再次安装

## 2 运行测试代码


- 以下为评估脚本

  - `scripts/helm/full.sh` (没有压缩)
  - `scripts/helm/h2o.sh` （h2o算法）
  - `scripts/helm/local.sh`（直接保留和h2o算法相同数量token kv值的结果）

- 这里第一个参数为数据集名字，第二个参数为模型在huggface中的名字，第三个参数为模型名字

- 下面是一个运行H2O测试程序的示例：

  > bash scripts/helm/h2o.sh xsum huggyllama/llama-7b llama

- **注意**：在运行前请修改`run_helm.py`中150-152行代码中的第一个参数，改为模型路径


## 3 用helm框架对生成结果进行评估

### 3.1 下载helm评估框架所需数据

在/root 下新建.py文件，写入如下程序，并运行：

```
import nltk
nltk.download("wordnet")
nltk.download("punkt_tab")
```

将默认下载到 /root/nltk_data，这也是nltk默认搜索路径

也可以直接将我们提供的nltk数据放在`/root`目录下

### 3.2 完善运行环境

在 h2o_hf/helm 下新建 prod_env 与 prod_env/cache 文件夹

登录网站（https://www.together.ai/） 注册账号，复制api key

在 h2o_hf/helm/prod_env 文件夹下新建文件 credentials.conf 填入

```
{
    togetherApiKey: "<your api key>"
}
```

### 3.3 进行评估

- 注：首先你应当已经按照步骤一的指示通过 `bash scripts/helm/h2o.sh xsum huggyllama/llama-7b llama` 等得到了生成结果，它会保存在.jsonl文件里

通过`cd ./helm`进入helm文件夹下，运行`./our_run.sh`，其内容如下：

```
jsonl=../xsum-llama-full.jsonl # 我们假设此前生成的结果文件在 h2o_hf 下，你可以更改此路径
task=xsum
model_arch=llama
output_name=xsum-llama-test # 仅控制评估指标的输出路径，可自行更改

# 将此前生成的结果文件与 任务数据集的request 打包，整理为适当格式
python scripts/offline_eval/import_results.py together ${jsonl} --cache-dir prod_env/cache 

# 执行第三方库的 helm-run 指令进行评估
helm-run --conf src/helm/benchmark/presentation/${task}/run_specs_${model_arch}.conf --local --max-eval-instances 1000 --num-train-trials=1 --suite ${output_name} -n 1

# 执行 helm-summarize 指令汇总结果
helm-summarize --suite ${output_name}
```

评估时如果出现网络问题可以尝试使用代理

```
export HF_ENDPOINT=https://hf-mirror.com
```

评估结果在 `h2o_hf/helm/benchmark_output/runs/<output_name>` 文件夹中，对于上面的例子我们可以在 `h2o_hf/helm/benchmark_output/runs/xsum-llama-test/runs.json` 中找到

```
    ...
    "stats": [
      {
        "name": {
          "name": "rouge_1",
          "split": "test"
        },
        "count": 1,
        "sum": 0.36467365185904915,
        "sum_squared": 0.13298687236021497,
        "min": 0.36467365185904915,
        "max": 0.36467365185904915,
        "mean": 0.36467365185904915,
        "variance": 0.0,
        "stddev": 0.0
      },
      {
        "name": {
          "name": "rouge_2",
          "split": "test"
        },
        "count": 1,
        "sum": 0.14136855201047124,
        "sum_squared": 0.019985067497537312,
        "min": 0.14136855201047124,
        "max": 0.14136855201047124,
        "mean": 0.14136855201047124,
        "variance": 0.0,
        "stddev": 0.0
      },
      {
        "name": {
          "name": "rouge_l",
          "split": "test"
        },
        "count": 1,
        "sum": 0.28649672075972754,
        "sum_squared": 0.0820803710060773,
        "min": 0.28649672075972754,
        "max": 0.28649672075972754,
        "mean": 0.28649672075972754,
        "variance": 0.0,
        "stddev": 0.0
      },
      {
        "name": {
          "name": "summarization_coverage",
          "split": "test"
        },
        "count": 1,
        "sum": 0.7980508335593463,
        "sum_squared": 0.6368851329447673,
        "min": 0.7980508335593463,
        "max": 0.7980508335593463,
        "mean": 0.7980508335593463,
        "variance": 0.0,
        "stddev": 0.0
      },
      ...
```

这对应于H2O论文中 XSUM 任务的评估指标

可以在 `./helm` 下运行 `helm-server` 查看可视化评估指标页面 
