env_tag=$1
scriptdir="$(dirname "$0")"
docker_tag=$(echo -n $scriptdir/base.Dockerfile | sha256sum | awk '{print $1}')
docker build -t "tm_ci:$env_tag" -f $scriptdir/ci.Dockerfile --build-arg TAG="$docker_tag" .