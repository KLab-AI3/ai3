from . import run_conv2d, run_mha, run
import sys

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == 'conv2d':
            run_conv2d()
        elif arg == 'mha':
            run_mha()
        else:
            print(f'Invalid op {arg}')
else:
    run()
