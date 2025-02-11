
from src.base import BaseMessage, DataElem
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def read_sol(data: dict) -> dict | None:
    
    return DataElem(
        elem=[
            BaseMessage(speaker="user", message=data["problem"]),
            BaseMessage(speaker="assistant", message=data["solution"])
        ]
    ).model_dump()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def format_conversation(data: DataElem[BaseMessage]) -> dict | None:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": data.elem[0].message},
        ],
        "sol": parse(
            data.elem[1].message,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    }

