from src.data import DataModule
from src.inference import InferenceConfig
from src.model import ModelConfig
from src.train import TrainConfig
from src.tokenizer import TokenizerConfig
from src.env import MODEL_SAVE_DIR
import os
def main(config: InferenceConfig):
    # load config from model_path
    model_name = config.pretrained_model
    if os.path.exists(os.path.join(MODEL_SAVE_DIR, model_name, "config.yaml")):
        c = TrainConfig.load(os.path.join(model_name, "config.yaml"))
        model_config = c.model_load_config
        tokenizer_config = c.tokenizer_config
    else:
        model_config = ModelConfig(base_config=model_name)
        tokenizer_config = TokenizerConfig(from_pretrained=model_name)
    
    model = BaseModel(model_config, inference_config=config)

    model.setup("inference")

    # load data from config
    datamodule = DataModule(
        config.data_loader_config,
        config.data_processor_config,
        tokenizer_config
    )

    datamodule.prepare_data()
    datamodule.setup(["train", "dev"]) # type: ignore


    model.model.generate(**config.generation_config.model_dump())