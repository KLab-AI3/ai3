from . import backward, compile, opcheck, train, run
import sys

if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        if arg == 'conv2d':
            opcheck.conv2d()
            compile.conv2d()
            backward.conv2d()
            train.conv2d()
        else:
            print(f'Invalid op {arg}')
else:
    run()
