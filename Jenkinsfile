def show_node_info() {
    sh """
        echo "NODE_NAME = \$NODE_NAME" || true
        lsb_release -sd || true
        uname -r || true
        cat /sys/module/amdgpu/version || true
        ls /opt/ -la || true
    """
}

def runTests() {
    // def targetNode = "${arch}"
    def targetNode = "aus-navi3x-08.amd.com"
    echo "The value of targetNode is: ${targetNode}"

    node(targetNode) {
        show_node_info()
        checkout scm

        sh """
        docker build -t tm_ci:${env.BUILD_ID} .
        docker run --rm --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=/home/jenkins:/home/jenkins torch_migraphx_ci bash -c "pip install transformers ; cd /workspace/torch_migraphx/tests/ ; pytest"
        """
    }
}

pipeline {
    agent { label 'build-only' }
    environment {
        MAIN_BRANCH = 'master'
    }
    stages {
        stage('matrix') {
            matrix {
                axes {
                    axis {
                        name 'arch'
                        values 'gfx1100'
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