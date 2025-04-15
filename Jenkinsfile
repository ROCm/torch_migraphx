def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME" || true
        lsb_release -sd || true
        uname -r || true
        cat /sys/module/amdgpu/version || true
        ls /opt/ -la || true
    """
}

def get_nodes(arch){
    if(arch == 'MI'){
        return "migraphx && gfx90a"
    } else if(arch == 'Navi') {
        return "migraphx && gfx1101"
    } else {
        return "migraphx"
    }
}

def runTests() {
    def targetNode = get_nodes("${arch}")
    echo "The value of targetNode is: ${targetNode}"

    node(targetNode) {
        show_node_info()
        checkout scm

        gitStatusWrapper(credentialsId: "${env.status_wrapper_creds}", gitHubContext: "Jenkins - pytest-${arch}", account: 'ROCmSoftwarePlatform', repo: 'torch_migraphx') {
            sh '''
            docker_tag=$(sha256sum ./ci/base.Dockerfile | awk '{print $1}')
            docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=`pwd`:/workspace/torch_migraphx rocm/torch-migraphx-ci-ubuntu:$docker_tag bash -c \
            'cd /workspace/torch_migraphx/py ; python -m pip install . ; cd /workspace/torch_migraphx/tests/ ; pytest'
            '''
        }
    }
}

pipeline {
    agent { label 'build-only' }
    stages {
        stage('matrix') {
            matrix {
                axes {
                    axis {
                        name 'arch'
                        //values 'MI', 'Navi'
                        values 'Any'
                    }
                }
                stages {
                    stage('unit-tests'){
                        steps {
                            script {
                                runTests()
                            }
                        }
                    }
                }
            }
        }
    }
}
