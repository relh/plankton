import os
OUT_OF = 4
I_AM_THE = 0
# 0 indexed!

mine = sorted(os.listdir('test'))[I_AM_THE::OUT_OF] + ['end\n']
open('filelist.txt','w').write('\n'.join(mine))
