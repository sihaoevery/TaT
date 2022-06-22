#! /bin/bash
#NOTE: the argument 'adjust_lr' is set to False

NUM_GPUS=8
python -m torch.distributed.launch \
	--nproc_per_node=${NUM_GPUS} \
	--use_env examples/image_classification.py \
	--world_size ${NUM_GPUS} \
	--log ./result/ilsvrc2012/tat/resnet18_from_resnet34.txt \
	-adjust_lr \
	--config configs/sample/ilsvrc2012/single_stage/tat/resnet18_from_resnet34_attn.yaml