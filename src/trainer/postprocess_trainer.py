from trl import DataCollatorForCompletionOnlyLM, GRPOTrainer
from transformers import PreTrainedModel, PreTrainedTokenizer


def SFTTrainer(**kwargs):
    """SFT 학습을 할 경우 target만 학습하기 위한 코드"""
    model: PreTrainedModel = kwargs["model"]
    if model.config.model_type == "llama":
        response_template = "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
    elif model.config.model_type == "qwen2":
        response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=kwargs["processing_class"]
    )

    kwargs["data_collator"] = collator

    return kwargs