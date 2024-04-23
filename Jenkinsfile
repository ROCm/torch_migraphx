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
    echo "The value of targetNode is: ${targetNode}"

    node(targetNode) {
        show_node_info()
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
                        values 'MI250', 'gfx1100'
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