
# VO-Digger -- Virtual Observatory Digger

Tool for classification data using deep neural networks on GPU (CUDA).

## Installation

So far it is enough to compile
```
cmake [-DCPU_ONLY=ON] .
make
```
This series of commands will produce a binary vodigger

Use the constant CPU_ONLY if you don't have a CUDA GPU (or CUDA libraries) on youur computer.

## Usage

The application is configurable via JSON files with sections designated to surrounding server. We
provide a commandline interface with few arguments.
```
./vodigger --train <repository>
./vodigger --test <model-snapshot> <repository>
./vodigger --time <repository>
./vodigger --dump <model-snapshot> <repository>
```
The mandatory argument for all parameters is the repository in which the classifier operates. It has to
be a folder with a configuration file named \texttt{config.json} by default.


### config.json

```
data : {
    "name":"stellar_spectra",
    "params":
    {
        "mode": "GPU",
        "solver": "solver.prototxt",
        "model": "model1.prototxt",
        "test_iters": 1,
        "bench_iters": 15
    }
}
```

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
