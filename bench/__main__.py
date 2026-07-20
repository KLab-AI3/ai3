from bench import layer, swap_conv2d, swap_mha, convert
import model_zoo

layer.run()
model_zoo.run_on(swap_conv2d.runner)
model_zoo.run_on(swap_mha.runner, op=model_zoo.MHA)
model_zoo.run_on(convert.runner)
