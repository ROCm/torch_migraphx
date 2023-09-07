TAG=${1:-torch_migraphx}
DOCKERFILE=${2:-Dockerfile}
GPU_ARCH="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')"
docker build --no-cache -t $TAG -f $DOCKERFILE --build-arg GPU_ARCH="$GPU_ARCH" .