#!/bin/bash
pip install transformers==4.47.0
pip install accelerate==1.1.1
pip install deepspeed==0.15.4
pip install trl==0.13
accelerate launch --config_file example/accelerate/deepspeed3.yaml run.py ${@:1}