from test import unit, convert, swap_conv2d, swap_mha, ops
import model_zoo

unit.run()
model_zoo.run_on(swap_conv2d.runner)
model_zoo.run_on(swap_mha.runner, op=model_zoo.MHA)
model_zoo.run_on(convert.runner)
model_zoo.run_on(convert.runner, op=model_zoo.MHA)
ops.run()
