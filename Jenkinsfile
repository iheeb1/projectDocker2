pipeline {
    agent any

    environment {
        PYTHON_VERSION = '3.8'  
    }

    stages {
        stage('Build') {
            steps {
                script {
                    bat "docker-compose build"
                }
            }
        }

        stage("Test") {
            steps {
                script {
                    bat "py unittests.py"
                }
            }
            post {
                always {
                    junit 'test-reports/*.xml'
                }
            }
        }

        stage("Run") {
            steps {
                script {
                    bat "docker-compose up"
                }
            }
        }
    }
}
