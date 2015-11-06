
# VO-Digger -- Virtual Observatory Digger

Tool for classification data using deep neural networks on GPU (CUDA).

## Installation

So far it is enough to compile
```
cmake [-DCPU_ONLY=ON] .
make
```
This series of commands will produce a binary vodigger

Use the constant CPU_ONLY if you don't have a CUDA GPU (or CUDA libraries) on your computer.

## Usage

The application is configurable via JSON file. A complementary commandline interface with few
arguments is provided also.

```
./vodigger --train <repository>
./vodigger --test <model-snapshot> <repository>
./vodigger --time <repository>
./vodigger --dump <model-snapshot> <repository>
./vodigger --guess <model-snapshot> <repository>
```

The mandatory argument for all parameters is the repository in which the classifier operates. It has
to be a folder with a configuration file named `config.json` by default. The config file  replaces
caffe's `solver.protobuf` file and has the same variables under section `"train"`.


### config.json

```
{
    "name":"stellar_spectra",
    "data":
    {
        "train": {
            "file" : "train_data.csv",
            "start": 1,
            "end": 1863,
            "label": 0,
            "chunk_size": 0.8
        },
        "test": {
            "file": "test_data.csv",
            "start": 1,
            "end": 1863,
            "label": 0,
            "chunk_size": 0.5
        }
        "guess": {
            "file": "unknown_data.csv",
            "start": 1,
            "end": 1863,
            "id": 0,
            "chunk_size": 0.2
        }

    }
    "params":
    {
        "mode": "GPU",
        "model": "model1.prototxt",
        "solver": "SGD",         /* optional, "SGD" is default value */
        "train": {
            "iter": 200,         /* how many times will be input layer filled with new data */
            "base_lr": 0.1,
            "lr_policy": "step", /* learning rate policy: "fixed": keep lr, "step" drop the lr */
            "stepsize": 150,     /* learning rate drops every # iterations */
            "gamma": 0.90,       /* learning rate gets multiplied by # every `stepsize` */
            "momentum": 0.1,     /* parameters change will have momentum from previous iteration */

            "display":       210, /*  display stats every # of iterations */
            "test_interval": 210, /*  launch testing of trained net every # iterations */
            "test_iter":     1,

            "snapshot": 500
            /* snapshot_after_train:  true /* is true by default */
        },
        "test": {
            "iter": 1
        }
        "benchmark": {
            "iter": 15
        }
    }
}
```
 + __params__ -- specifies parameters of solver. The model definition is in a separate file
    + __mode__ [string] "GPU", "CPU" -- the same as caffe's `mode` parameter - denotes device to run the
    network on
    + __train__ [object] -- all parameters will be used as caffe's SOLVER parameters. For more details
    please refer to [caffe's documentation](http://caffe.berkeleyvision.org/tutorial/solver.html)
        + __iter__ [number] -- Number of times of calling FORWARD function of a neural net. Note
        that in case of a  large file and small `batch_size` even many iterations doesn't guarantee usage of
        the whole file.


The parameter `model` and `solver` are very similar. They contain names of a model and
solver definition files, respectively. Both files has to be proto files following the requirements for Caffe model and solver definition, respectively. There are
two differences. First is, that we set up the running mode (either CPU or GPU) directly in this
config file (by `params.mode`) and not in a solver. The reason is simply that the solver
is not always needed and therefore it should not be responsibility of a solver to set up the mode.
Second difference is that we have two test_iteration values -- one in solver and second in config
file. The one in solver says number of test iterations while training a network (for validation).
The test_iter parameter in the config file determines a number of iterations when the
classifier is running in `--test` mode. If the networks outputs ArgMax layer in testing phase
the classifier creates a confusion matrix out of it. Last parameter bench_size denotes how
many iterations of forward-backward pass (in training mode) should be done in order to benchmark
given model.
