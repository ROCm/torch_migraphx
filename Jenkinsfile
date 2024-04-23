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
    def targetNode = "${arch}"
    echo "The value of targetNode is: ${targetNode}, gpu arch is: ${GPU_ARCH}"

    node(targetNode) {
        show_node_info()
        checkout scm

        def testImage = docker.build("tm_test:${env.BUILD_ID}", "--build-arg GPU_ARCH=${GPU_ARCH}")
        testImage.withRun('--network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v=/home/jenkins:/home/jenkins'){
            sh 'python -c "import torch; import torch_migraphx; import migraphx"'
        }
    }
}

pipeline {
    agent { label 'build-only' }
    environment {
        MAIN_BRANCH = 'master'
        GPU_ARCH = sh(script: 'GPU_ARCH="$(/opt/rocm/bin/rocminfo | grep -o -m 1 \'gfx.*\')"', returnStdout: true)
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