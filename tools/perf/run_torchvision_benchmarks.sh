#!/bin/bash

resultsfile=./perf.out

while read Models Batch Params
do
	model_params=$(echo $Params | tr -d '""')

	if [ -z "$model_params" ]; then Precision=$(echo "fp32")
    else Precision=$(echo $model_params | sed s/--//g)
    fi

    echo "Model:" $Models, "Batch:" $Batch, "Precision:" $Precision 2>&1 | tee -a $resultsfile
	python3 benchmark_torchvision.py -m $Models -b $Batch $model_params 2>&1 | tee -a $resultsfile
done <<FILELIST
alexnet 1
alexnet 1 "--fp16"
resnet50 1
resnet50 1 "--fp16"
resnet50 2
resnet50 2 "--fp16"
resnet50 4
resnet50 4 "--fp16"
resnet50 8
resnet50 8 "--fp16"
resnet50 16
resnet50 16 "--fp16"
resnet50 32
resnet50 32 "--fp16"
resnet50 64
resnet50 64 "--fp16"
inception_v3 1
inception_v3 1 "--fp16"
mobilenet_v3_small 1
mobilenet_v3_small 1 "--fp16"
wide_resnet50_2 1
wide_resnet50_2 1 "--fp16"
vgg16 1
vgg16 1 "--fp16"
mobilenet_v3_large 1
mobilenet_v3_large 1 "--fp16"
efficientnet_b4 1
efficientnet_b4 1 "--fp16"
densenet169 1
densenet169 1 "--fp16"
FILELIST
