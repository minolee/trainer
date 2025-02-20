# 각종 환경 변수들 저장
# 직접 수정 또는 os.environ을 통해 override 가능

import os
import accelerate
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
MODEL_SAVE_DIR = os.environ.get("MODEL_SAVE_DIR", "rsc/model")
try:
    Accelerator = accelerate.Accelerator()

    print(Accelerator.state)
    # print(Accelerator.is_local_main_process, Accelerator.process_index) # 여기선 False와 index가 정상적으로 출력됨
except:
    import traceback
    # traceback.print_exc()
    print("Accelerator is not set")
    Accelerator = None
