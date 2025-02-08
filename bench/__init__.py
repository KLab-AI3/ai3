import time
from typing import Tuple
import ai3
import torch


def warm_up(runner, data):
    if isinstance(data, Tuple):
        input = (data[0][0], data[1][0], data[2][0])
        runner(*input)
    else:
        input = data[0]
        runner(input)


def timed_predict(runner, data, grad: bool):
    out = None
    start_time = None
    if isinstance(runner, torch.nn.Module):
        warm_up(runner, data)
        if grad:
            start_time = time.time()
            if isinstance(data, Tuple):
                out = runner(*data)
            else:
                out = runner(data)
        else:
            with torch.inference_mode():
                start_time = time.time()
                if isinstance(data, Tuple):
                    out = runner(*data)
                else:
                    out = runner(data)
    elif isinstance(runner, ai3.Model):
        warm_up(runner, data)
        start_time = time.time()
        out = runner.predict(
            data, out_type=torch.Tensor)
    else:
        assert False and f'invalid runner f{type(runner)}'
    end_time = time.time()
    assert (start_time > 0)
    latency = end_time - start_time

    return out, latency


def predict_show_time(runner, data, runner_name: str, grad: bool = False):
    out, latency = timed_predict(runner, data, grad=grad)
    print(f'  Time {runner_name}: {latency} seconds')
    return out
