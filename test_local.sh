#! /bin/bash

NUM_GPUS=8
python -m torch.distributed.launch \
	--nproc_per_node=${NUM_GPUS} \
	--use_env examples/image_classification.py \
	--world_size ${NUM_GPUS} \
	-test_only \
	--config configs/sample/ilsvrc2012/single_stage/tat/resnet18_from_resnet34_attn.yaml