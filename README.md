# MLOps Project

## Introduction
모든 과정을 config를 사용하여 관리하고, 외부 요인의 변화가 없을 시 reproduce 가능한 상태로 저장하는 것을 목표로 개발

## Requirements
python >= 3.10

## Run
`python run.py --mode [train | inference | evaluation] --config <config_path>`

## Deeper inside
데이터 준비, 모델 준비, 학습/추론/평가 준비 -> 실행 의 과정으로 이루어져 있습니다.

각각의 과정은 모두 Config로 제어 가능합니다. Config class를 참고해서 작성해 주세요.

TrainConfig, InferenceConfig, EvaluationConfig는 모두 다른 형식을 가지고 있습니다.
이는 각각의 과정을 구분하여 실험 구분이 쉽도록 하기 위한 목적입니다.

## 공통과정
### Data 준비

## 학습 준비

## Inference 준비

## Evaluation 준비

## TODO LIST
* Interactive

## Author
- Minho Lee