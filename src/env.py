# 각종 환경 변수들 저장
# 직접 수정 또는 os.environ을 통해 override 가능

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
MODEL_SAVE_DIR = "rsc/model"
INFERENCE_SAVE_DIR = "rsc/inference"
EVALUATION_SAVE_DIR = "rsc/evaluation"