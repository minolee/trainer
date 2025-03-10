# Trainer Project

## Introduction
모든 과정을 config를 사용하여 관리하고, 외부 요인의 변화가 없을 시 reproduce 가능한 상태로 저장하는 것을 목표로 개발하였습니다.

Core logic을 최대한 건드리지 않는 대신 custom function을 사용 가능하도록 만들고,
custom function을 사용하여 개발 시 해당 파일의 tracking이 가능하도록 만들었습니다.

-> 실제 스크립트를 실행할 당시의 코드를 항상 저장합니다.

## Requirements
`python >= 3.10`

* Deepspeed 사용시 `python==3.10`으로 세팅해야 합니다.

## How to run
`python run.py --run_config <config_path>`

`python run.py --help` 를 치면 사용 방법을 알 수 있습니다.

### Accelerate
`python run.py --run_config <config_path> --accelerate_config <accelerate_config_path>`

자동으로 accelerate launch를 수행합니다.

### multi-node (beta)
`python run.py --run_config <config_path> --accelerate_config <accelerate_config_path> --nodes [nodelist]`

slurm style node list를 전달 시 각각의 node에 ssh command를 수행하는 방식으로 multinode 학습을 진행합니다. 자동으로 accelerate config의 num_machines와 machine_rank를 수정합니다.

slurm을 통해 스크립트를 실행할 경우에도 적용됩니다. `sbatch --nodes=X ...` 과 같이 실행하면 자동으로 node를 분배해서 실행합니다.


### Deepspeed
미구현, 하지만 accelerate config에 deepspeed를 사용할 수 있습니다.

### Slurm
Slurm을 사용해서 실행하기 위해서는 OCI bundle 형태로 준비해야 합니다.

https://slurm.schedmd.com/containers.html#bundle 를 참고하여 bundle을 만들어 주세요.

Dockerfile을 사용해서 docker image를 만들면 됩니다.

```bash
docker build -t trainer .
enroot import -o [PATH] dockerd://trainer
```

## (필독) Config 작성 방법

데이터 준비, 모델 준비, trainer 준비 -> 실행 의 과정으로 이루어져 있습니다.

각각의 과정은 모두 Config로 제어 가능합니다. Config class를 참고해서 작성해 주세요.

예시 config 파일은 [example/base](https://github.com/minolee/trainer/tree/main/example/train/base) 디렉토리에서 확인할 수 있습니다.


### Data 준비

config yaml 파일에 `data` 부분에 정의합니다.

TRL의 각 trainer마다 지원하는 형식이 있습니다.

여기서는 Raw data를 각각의 형식으로 가공하기 위해 아래 과정을 따릅니다.
Raw data -> list of BaseMessage -> TRL-supported format

[BaseMessage](https://github.com/minolee/trainer/tree/main/src/base/base_message.py)는 speaker와 message로 이루어져 있습니다. 이는 TRL에서 사용하는 chat template과 유사하면서도, 사내에서 구축한 format을 사용하기 편하도록 만들기 위함입니다.

필수 정의
* source: raw data path
* reader: raw data를 Instance(list of BaseMessage)로 변환하는 함수를 정의합니다.
* formatter: Instance(list of BaseMessage)를 trl trainer에서 지원하는 형식으로 변환하는 함수를 정의합니다.

함수 이름을 제공하면 자동으로 찾아서 등록합니다.

이외 config에 필요한 key, value type은 [DataConfig](https://github.com/minolee/trainer/tree/main/src/data/config.py)에 정의되어 있습니다.

```yaml
data:
  sources:
    - source: rsc/data/preference/processed/dpo_1cycle_241016.jsonl # 로컬 파일에서 읽어옵니다
      split: train # 이 파일을 train split으로 정의합니다
      limit: 500 # 이 파일에서 맨 앞 500개만 사용합니다.
      reader: read_preference # read_preference 함수를 사용하여 json instance를 BaseMessage 형태로 가공합니다.
    - source: AI-MO/NuminaMath-TIR # hf data hub에서 불러옵니다.
      use_cache: true # cache화합니다. (default false)
      reader: reader.read_sol # custom file reader.py의 read_sol 함수를 사용하여 data instance를 BaseMessage 형태로 가공합니다.
```

config file과 같은 디렉토리에 있는 파이썬 파일에 있는 함수를 사용할 수 있습니다.

#### Issue
* Chat template의 경우 model에 정의된 jinja template를 사용합니다. [HF document](https://huggingface.co/docs/transformers/chat_templating)
  * default chat template를 사용하는 경우 system, user, assistant 가 아닌 다른 role을 사용할 때 오류가 나고 있어, 현재 reader function에서는 role을 강제로 변경하고 있습니다.


### 모델 준비

config yaml 파일의 `model` 부분에 정의합니다.

```yaml
model:
  path: Qwen/Qwen2-0.5B-Instruct # local path or hf model card
```

### Trainer 준비

config yaml 파일의 `trainer` 부분에 정의합니다.

```yaml
trainer: 
  name: SFTTrainer # trl에 정의된 trainer 이름. 또는 custom trainer를 정의할 수 있습니다.
  # 이 아래 부분은 Trainer의 argument로 들어갑니다.
  num_train_epochs: 1
  gradient_accumulation_steps: 4
  per_device_train_batch_size: 8
  logging_steps: 5
```

### Custom code integration

필요한 class가 없는 경우나 특수한 로직이 필요한 경우, custom code를 작성하여 이를 사용할 수 있습니다.

config yaml 파일과 같은 디렉토리에 파이썬 파일을 작성하면, 자동으로 해당 파이썬 파일을 가져와서 함수 이름을 불러옵니다.

custom 함수를 사용하기 위해서는 <파일명>.<함수명> 으로 불러오면 됩니다.
[open-r1 example](https://github.com/minolee/trainer/tree/main/example/train/custom/open_r1) 안의 예시 파일들을 확인해 주세요.


### Document
https://minolee.github.io/trainer/

### Code Concept
목적이 있는 모든 config는 callable입니다. config를 로드한 뒤 call하면 목적에 맞는 object를 반환하거나(예시: DatasetConfig), 프로세스를 실행합니다(예시: TrainConfig).

따라서, config를 새로 만드는 경우는 완전히 새로운 process가 필요한 것이라고 생각하면 됩니다. 기존 process를 개선하거나 기능을 추가하고 싶은 경우에는 해당 process를 상속하는 subclass를 만든 뒤, 필요한 추가 기능을 구현하고 config에 동작 제어를 추가해 주세요.


## TODO LIST
우선순위별 정리

* Callbacks
* Peft (완료)
* RL (완료)
  * DPO (완료)
  * GRPO (진행중)

## Author
- Minho Lee
