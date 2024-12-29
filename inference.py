from src.inference import InferenceConfig
from src.train import TrainConfig
from src.env import MODEL_SAVE_DIR
import os
def main(config: InferenceConfig):
    # load config from model_path
    model_name = config.pretrained_model
    if os.path.exists(os.path.join(MODEL_SAVE_DIR, model_name, "config.yaml")):
        config = InferenceConfig.load(os.path.join(model_name, "config.yaml"))
    # train_config = 