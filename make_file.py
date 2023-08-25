from pathlib import Path
import os
from tqdm import tqdm
ROOT = 'H:\LJQ\VOC'
file = 'VOC2007_test.txt'
file_new = 'VOC2007_test_new.txt'
with open(Path(ROOT)/file,'r') as f:
    contant = f.readlines()
contant_new = [str(c).replace('E:\models\Few-shot_OD\dataset','H:\LJQ') for c in contant]
print(contant_new)
with open(Path(ROOT)/file_new,'w') as f:
    f.writelines(contant_new)