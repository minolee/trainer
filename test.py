from transformers import AutoModel
import torch
model = AutoModel.from_pretrained("bert-base-uncased")

torch.save(model.state_dict(), "rsc/model/test/model.bin")

