Models="resnet50 vgg16_bn alexnet densenet169 efficientnet_b6 googlenet mnasnet1_0 mobilenet_v3_large regnet_y_32gf shufflenet_v2_x1_5 squeezenet1_1 convnext_base inception_v3"

for mod in $Models; do
    python3 benchmark_torchvision.py -m $mod --fp16
done

for mod in $Models; do
    python3 benchmark_torchvision.py -m $mod
done