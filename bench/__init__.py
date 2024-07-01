from collections import defaultdict
import time
import ai3
import torch
from typing import Optional

USE_TORCH_COMPILE = False


def warm_up(runner, data):
    data_batch = None
    if data.dim() == 4:
        data_batch = data[0]
    elif data.dim() == 3:
        data_batch = data
    assert (data_batch is not None)
    runner(data_batch)


def predict_show_time(runner, data, runner_name: str, store: Optional[defaultdict[str, float]] = None, recur: bool = True):
    out = None
    start_time = -1
    if isinstance(runner, torch.nn.Module):
        warm_up(runner, data)
        with torch.inference_mode():
            start_time = time.time()
            out = runner(data)
    elif isinstance(runner, ai3.Model):
        warm_up(runner, data)
        start_time = time.time()
        out = runner.predict(data, out_type=torch.Tensor)
    else:
        print(f"invalid runner f{type(runner)}")
        assert (False)
    end_time = time.time()
    assert (start_time > 0)
    inference_time = end_time - start_time
    print(f"  Time {runner_name}: {inference_time} seconds")

    store_exist = store is not None
    if store_exist:
        store[runner_name] = inference_time
    if USE_TORCH_COMPILE and isinstance(runner, torch.nn.Module) and recur:
        predict_show_time(torch.compile(runner), data,
                          runner_name + " compiled", recur=False)
    return out
