# tfserving_sample

## Dir
```
ubuntu@<ip-addr>:~/mycode/ml/tensorflow$ ls
models  serving  tfserving_golang  tfserving_sample
```

The tensorflow serving is from:
```
git clone https://github.com/tensorflow/serving.git
```

## git diff
```
ubuntu@<ip-addr>:~/mycode/ml/tensorflow/serving/tensorflow_serving/tools/docker$ git branch
  master
* r1.12
ubuntu@<ip-addr>:~/mycode/ml/tensorflow/serving$ git diff
diff --git a/tensorflow_serving/tools/docker/Dockerfile.devel b/tensorflow_serving/tools/docker/Dockerfile.devel
index 6014159..24315cc 100644
--- a/tensorflow_serving/tools/docker/Dockerfile.devel
+++ b/tensorflow_serving/tools/docker/Dockerfile.devel
@@ -13,7 +13,7 @@
 # limitations under the License.
 FROM ubuntu:16.04 as base_build
 
-ARG TF_SERVING_VERSION_GIT_BRANCH=master
+ARG TF_SERVING_VERSION_GIT_BRANCH=r1.12
 ARG TF_SERVING_VERSION_GIT_COMMIT=head
 
 LABEL maintainer=gvasudevan@google.com
diff --git a/tools/run_in_docker.sh b/tools/run_in_docker.sh
index 4b9f05a..ee7f380 100755
--- a/tools/run_in_docker.sh
+++ b/tools/run_in_docker.sh
@@ -70,7 +70,8 @@ function get_python_cmd() {
 (( $# < 1 )) && usage
 [[ "$1" = "-"* ]] && [[ "$1" != "-d" ]] && usage
 
-IMAGE="tensorflow/serving:nightly-devel"
+# IMAGE="tensorflow/serving:nightly-devel"
+IMAGE="$USER/tf-devel"
 [[ "$1" = "-d" ]] && IMAGE=$2 && shift 2 || true
 [[ "${IMAGE}" = "" ]] && usage
 
@@ -91,9 +92,9 @@ fi
 [[ ! -x $(command -v docker) ]] && echo "ERROR: 'docker' command missing from PATH." && usage
 
 echo "== Pulling docker image: ${IMAGE}"
-if ! docker pull ${IMAGE} ; then
+if ! sudo docker pull ${IMAGE} ; then
   echo "WARNING: Failed to docker pull image ${IMAGE}"
 fi
 
 echo "== Running cmd: ${CMD}"
-docker run ${RUN_OPTS[@]} ${IMAGE} bash -c "$(get_switch_user_cmd) ${CMD}"
+sudo docker run ${RUN_OPTS[@]} ${IMAGE} bash -c "$(get_switch_user_cmd) ${CMD}"
```

## Build docker
build your own docker, with `tensorflow_model_server` ready
```
ubuntu@<ip-addr>:~/mycode/ml/tensorflow/serving/tensorflow_serving/tools/docker$ sudo docker build --pull -t $USER/tf-devel -f Dockerfile.devel .
```
This step may take more than 30 minutes, and costs lots of memory

## Train a model
Follow the instruction in https://github.com/lenjoy/tfserving_sample/blob/master/train/train.py

Remember to download the training data by python script, and update the input path for training.

The model training can be done anywhere. Once it's done, you will get
```
ubuntu@<ip-addr>:~/mycode/ml/tensorflow/tfserving_sample$ ls /tmp/census_model/serving_savemodel/1542868254/
saved_model.pb  variables
```

## Start the server
### Option 1: Log into container, then start server
```
sudo docker run -it -p 8500:8500 \
--mount type=bind,source=/tmp/census_model/serving_savemodel,target=/models/census_model/serving_savemodel \
-v ~/mycode/ml/tensorflow/tfserving_sample:/src/tfserving_sample $USER/tf-devel
```

```
tensorflow_model_server --port=8500 --model_name=wide_and_deep --model_base_path=/models/census_model/serving_savemodel/ &
```

### Option 2: Start the container with the server running
```
sudo docker run -p 8500:8500 \
--mount type=bind,source=/tmp/census_model/serving_savemodel,target=/models/census_model/serving_savemodel \
-t $USER/tf-devel \
tensorflow_model_server --port=8500 --model_name=wide_and_deep --model_base_path=/models/census_model/serving_savemodel
```

## Test with client
### Python client
```
ubuntu@<ip-addr>:~/mycode/ml/tensorflow/tfserving_sample$ ../serving/tools/run_in_docker.sh python py_client/client.py --server=127.0.0.1:8500
```

### Golang client
Follow the instruction in https://github.com/lenjoy/tfserving_sample/blob/master/go_client/client.go
