import pandas as pd
import numpy as np

from utilities import sortTools
from pathlib import Path

st = sortTools()
folder = '/Users/gketawal/PycharmProjects/InternalTool/r0484-snr5pix2'
files = Path(folder).glob('*cxi')

# for file in files:
#     st.advanceSortFrames(file,'4')

df1 = pd.read_csv("badEvents-advanceSearch-r0484_1.list",delimiter=' ')
df1.columns = ['FileName', 'EventNumber', 'InflectionPoint1', 'InflectionPoint2']
print(df1)



