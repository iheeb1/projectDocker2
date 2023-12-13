node {
   stage('Build') {
       
         sh "docker-compose build"
   }
   stage("Test"){
        sh "python unittests.py"
        post {
                always {junit 'test-reports/*.xml'}
            }
    }

    stage("Run"){
        sh "docker-compose up "
    }
}
