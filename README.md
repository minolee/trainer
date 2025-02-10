# MLOps Project

## Introduction
모든 과정을 config를 사용하여 관리하고, 외부 요인의 변화가 없을 시 reproduce 가능한 상태로 저장하는 것을 목표로 개발하였습니다.

## Requirements
`python >= 3.10`

* Deepspeed 사용시 `python==3.10`으로 세팅해야 합니다.

# How to run
`python run.py --mode [train | inference | evaluation] --run_config <config_path>`

`python run.py --help` 를 치면 사용 방법을 알 수 있습니다.

### Accelerate
`python run.py --mode [train | inference | evaluation] --run_config <config_path> --accelerate_config <accelerate_config_path>`

자동으로 accelerate launch를 수행합니다.

### multi-node (beta)
`python run.py --mode [train | inference | evaluation] --run_config <config_path> --accelerate_config <accelerate_config_path> --nodes [nodelist]`

slurm style node list를 전달 시 각각의 node에 ssh command를 수행하는 방식으로 multinode 학습을 진행합니다.

이 때 반드시 accelerate config에 정의된 `num_machines`와 전달한 node 수가 같은지 확인해 주세요.

### Deepspeed
미구현, 하지만 accelerate config에 deepspeed를 사용할 수 있습니다.


## Deeper inside
데이터 준비, 모델 준비, 학습/추론/평가 준비 -> 실행 의 과정으로 이루어져 있습니다.

각각의 과정은 모두 Config로 제어 가능합니다. Config class를 참고해서 작성해 주세요.

TrainConfig, InferenceConfig, EvaluationConfig는 모두 다른 형식을 가지고 있습니다.
이는 각각의 과정을 구분하여 실험 구분이 쉽도록 하기 위한 목적입니다.

예시 config 파일은 [config/base](https://github.com/minolee/mlops/tree/main/config/base) 디렉토리에서 확인할 수 있습니다.

## 공통과정
### Data 준비

학습, 추론, 평가 모든 과정에서는 데이터를 읽어 와서 DataLoader 형태로 만드는 과정이 필요합니다.

여기서는 Raw data를 DataLoader화 하기 위해 아래 과정을 따릅니다.
Raw data -> list of BaseMessage -> Dataset -> DataLoader

#### [Reader](https://github.com/minolee/mlops/blob/main/src/data/reader/config.py)
Raw data를 중간 형태(list of BaseMessage) 형태로 만듭니다.

* sources: list of sources
  * name: 데이터셋 이름 (optional)
  * source: 데이터 파일 경로
  * split: train | dev | test | predict
  * limit: 데이터 수량 제한 (optional)
* reader: 데이터를 읽는 방법을 정의. [reader_fn](https://github.com/minolee/mlops/blob/main/src/data/reader/reader.py)에 정의된 함수를 사용할 것

#### [Formatter](https://github.com/minolee/mlops/blob/main/src/data/dataset/format_fn.py)
List of BaseMessage를 trainer에 맞는 형태로 가공합니다. 예를 들어, DPOTrainer의 경우 [preference dataset 형식](https://huggingface.co/docs/trl/dataset_formats#preference)으로 가공하는 과정입니다.


#### [DataLoader](https://github.com/minolee/mlops/blob/main/src/data/dataloader/config.py)
<strike>
Dataset을 받아 DataLoader를 만드는 과정을 제어합니다. 이 과정에서 batch_size, collate_fn, sampler 등을 정의할 수 있습니다.

* shuffle: 데이터셋 섞기 여부
* num_workers: dataloader worker
* batch_size: batch size


</strike>

250106 변경: Dataloader는 hf trainer의 argument로 넘기는 방식으로 변경하였음
250210 변경: HF Dataset으로 변경함. 중간 data 구조만 남김

### 모델 준비
모델은 3가지 로딩 방식이 있습니다.

* load from hub
* load from local
* load from scratch

## Develop

Task나 필요에 따라 추가적인 class를 정의해야 할 수 있습니다. 이 경우 참고해 주세요.

### Document
https://minolee.github.io/mlops/

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
* <strike>Evaluation 구현</strike> ([lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) 사용 권장)
* Interactive inference
* <strike>Dialogue Packing</strike> (SFTTrainer에 포함됨)

## Author
- Minho Lee