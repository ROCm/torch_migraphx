# WAN
Wan Model (Text to Video -- T2V) using MIGraphX optimizations on ROCm platform.

## Steps

To run the WAN, follow these steps below.

1) Install torch-migraphx.

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the inference, it will compile and run the model. 
NOTE: As of now, compiling the transformer only with torch-migraphx works. We are working on supporting compiling other parts of the model.

```bash
python3 wan.py --compile_migraphx=transformer --bf16 --deallocate
```

NOTE: If not using MI300X, enabling attention may be required:
```bash
MIGRAPHX_MLIR_USE_SPECIFIC_OPS="attention" python3 wan.py --compile_migraphx=transformer --bf16 --deallocate
```