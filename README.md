## ai3 - Algorithmic Innovations for Accelerated Implementations of *AI*

#### A Framework to Enable Algorithmic Design Choices in *DNNs*

### Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Implementing Custom Algorithms](#implementing-custom-algorithms)
4. [Testing](#testing)
5. [Benchmarks](#benchmarks)

This framework contains built-in high performance implementations of common deep learning operations and methods by which users can implement their own algorithms in *C++*. The framework’s built-in accelerated implementations yield outputs equivalent to and exhibit similar performance as implementations in *PyTorch*. The framework incurs no additional performance overhead, meaning that performance depends solely on the algorithms chosen by the user.

### Overview
The framework currently features two methods for algorithmic swapping. A function that swaps a specific module type and one that swaps every module type of a *DNN* and returns an object completely managed by the framework.

The first function called ```swap_backend```, takes the *DNN* and a mapping from the name of a module to an algorithmic selector, passing no algorithm for a module type is equivalent to passing *"default"*. The second function, implemented for convolution layers, ```swap_conv2d```, takes the *DNN* and an algorithmic selector for the module type in the function name. After ```swap_conv2d``` is executed, the model can still be used for training but will now use the selected algorithm for forward propagation. An example of this is seen in [example/train](./example/train.py).

There are three supported forms of algorithmic selection, the first is a string containing the name of the algorithm, this algorithm will be used for all modules of the associated type, the second is a list of algorithm names, as modules are encountered, they are replaced with an implementation of the algorithm in the list with the same index as that module has relative to the other modules of the same type, the third method for algorithmic selection is a function. This function is given the module to swap and returns the name of the algorithm to use, the function can optionally take the input size for this module if you pass a sample input shape to the swapping function.

#### Using String Selectors, Sampled from [example/manual_conv2d](./example/manual_conv2d.py)
```python
input_data = torch.randn(10, 3, 224, 224)
orig = ConvNet()
torch_out = orig(input_data)

model: ai3.Model = ai3.swap_backend(orig, {'conv2d': 'direct', 'maxpool2d': 'default'})
sb_out = model(input_data)
assert torch.allclose(torch_out, sb_out, atol=1e-6)

ai3.swap_conv2d(orig, ['direct', 'smm'])
sc_out = orig(input_data)
assert torch.allclose(torch_out, sc_out, atol=1e-6)
```

#### Function Selector, Sampled from [example/vgg16](./example/vgg16.py)
```python
def conv2d_selector(orig: torch.nn.Conv2d) -> str:
    in_channels = orig.weight.shape[1]
    if in_channels > 200:
        return 'smm'
    return 'direct'

vgg16 = torchvision.models.vgg16(
    weights=torchvision.models.VGG16_Weights.DEFAULT)
vgg16.eval()
torch_out = vgg16(input_data)

model: ai3.Model = ai3.swap_backend(vgg16, {'conv2d': conv2d_selector})
sb_out = model(input_data)
assert torch.allclose(torch_out, sb_out, atol=1e-4)
```
#### Function Selector With Input Shape, Sampled from [example/func_with_input_shape](./example/func_with_input_shape.py)
```python
def conv2d_selector(orig: torch.nn.Conv2d, input_shape: Sequence[int]) -> str:
    out_channels = orig.weight.shape[0]
    if out_channels < 50 and input_shape[0] < 50 and input_shape[1] > 150 and input_shape[2] > 150:
        return 'direct'
    return 'smm'

input_data = torch.randn(10, 3, 224, 224)
orig = ConvNet()
torch_out = orig(input_data)

model = ai3.swap_backend(
    orig, {'conv2d': conv2d_selector}, (3, 224, 224))
sb_out = model.predict(input_data)
assert torch.allclose(torch_out, sb_out.torch(), atol=1e-6)

ai3.swap_conv2d(orig, conv2d_selector, (3, 224, 224))
sc_out = orig(input_data)
assert torch.allclose(torch_out, sc_out, atol=1e-6)
```

### Installation
For users not seeking to implement their own algorithms the package will soon be on [pypi.org](pypi.org). For now the package can be installed via a link to this repository or the instructions below.
```sh
pip install git+<link to this repo>.git
```

For users who implemented their own algorithms as described below or users wishing to build from source, use *pip*.
```sh
pip install <path to local clone of this repository>
```

### Implementing Custom Algorithms
To implement custom algorithms, clone this repository and create an implementation for the desired operation declared in [src/ai3/custom](./src/ai3/custom). Set the boolean *constexpr* to *true* if you wish for the custom implementation to be used by default. Examples of how to implement algorithms are seen in the implementations at [src/ai3/csrc](./src/ai3/csrc)

### Testing
Testing can be done with *python* directly or through the [run](./run) script provided which uses *python*.

#### Unit Tests
```sh
# run all
./run utest
./run test.unit
python -m test.unit
# run a specific module
./run utest.conv2d
./run test.unit.conv2d
python -m test.unit.conv2d
```
The module can be any in [test/unit](./test/unit).

#### Swap Conv2D Tests
```sh
# run all
./run sctest
./run test.swap_conv2d
python -m test.swap_conv2d
# run a specific model
./run sctest.vgg16
./run test.swap_conv2d.vgg16
python -m test.swap_conv2d vgg16
```

#### Swap Backend Tests
```sh
# run all
./run sbtest
./run test.swap_backend
python -m test.swap_backend
# run a specific model
./run sbtest.vgg16
./run test.swap_backend.vgg16
python -m test.swap_conv2d vgg16
```

Models ran in both the *swap_conv2d* and *swap_backend* tests are in [runners/models.py](./runners/models.py)

### Benchmarks

Similar to testing, benchmarks can be done with *python* directly or through the [run](./run) script provided which uses *python*.

#### Layer Bench
```sh
# run all
./run lbench
./run bench.layers
python -m bench.layers
# run a specific module
./run lbench.conv2d
./run bench.layers.conv2d
python -m bench.layers.conv2d
```
The module can be any in [bench/layers](./bench/layers).

#### Swap Conv2D Bench
```sh
# run all
./run scbench
./run bench.swap_conv2d
python -m bench.swap_conv2d
# run a specific model
./run scbench.vgg16
./run bench.swap_conv2d.vgg16
python -m bench.swap_conv2d vgg16
```

#### Swap Backend Bench
```sh
# run all
./run sbbench
./run bench.swap_backend
python -m bench.swap_backend
# run a specific model
./run sbbench.vgg16
./run bench.swap_backend.vgg16
python -m bench.swap_backend vgg16
```

Models ran in both the *swap_conv2d* and *swap_backend* benchmarks are in [runners/models.py](./runners/models.py).
