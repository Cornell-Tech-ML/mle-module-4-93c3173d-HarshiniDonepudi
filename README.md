# MiniTorch Module 4

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module4.html

This module requires `fast_ops.py`, `cuda_ops.py`, `scalar.py`, `tensor_functions.py`, `tensor_data.py`, `tensor_ops.py`, `operators.py`, `module.py`, and `autodiff.py` from Module 3.


Additionally you will need to install and download the MNist library.

(On Mac, this may require installing the `wget` command)

```
pip install python-mnist
mnist_get_data.sh
```


* Tests:

```
python run_tests.py
```

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/fast_ops.py minitorch/cuda_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py project/run_fast_tensor.py project/parallel_check.py tests/test_tensor_general.py


## Results
Link to folder with output logs for both MNIST and sentiment analysis test on lr 0.01 and 0.05.
[Results](https://github.com/Cornell-Tech-ML/mle-module-4-93c3173d-HarshiniDonepudi/tree/master/project/output_logs)


### MNIST Training Logs

#### With learning rate 0.01 we get a final accuracy of 16/16
<img src="project/output_logs/mnist_0.01.1.png">


<img src="project/output_logs/mnist_0.01_25.png">

##### With learning rate 0.05 we get a final accuracy of 16/16
<img src="project/output_logs/mnist_0.05_1.png">


<img src="project/output_logs/mnist_0.05_25.png">

### Sentiment Training Logs


#### 79% Validation Accuracy with lr = 0.01

<img src="project/output_logs/sentiment_0.01_start.png">


<img src="project/output_logs/sentiment_0.01_end.png">


#### 79% Validation Accuracy with lr = 0.05
<img src="project/output_logs/sentiment_0.05_start.png">


<img src="project/output_logs/sentiment_0.05_end.png">
