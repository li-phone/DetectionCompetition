import os
import glob
import sys

epochs = glob.glob(os.path.join(sys.argv[1], 'epoch_*.pth'))
for e in epochs:
    if os.path.basename(e) != 'epoch_12.pth':
        print('remove {} ok!'.format(e))
        os.remove(e)
