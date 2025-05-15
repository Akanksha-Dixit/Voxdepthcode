import pandas as pd
import numpy as np

base = pd.read_csv("/home/yashashwee/cudaHello/drone/BaseDrone_red (12).txt")
meth = pd.read_csv("/home/yashashwee/cudaHello/drone/VDDrone_red (12).txt")


methX,methY,methZ = meth['x'].to_numpy(),meth['y'].to_numpy(),meth['z'].to_numpy()

baseX,baseY,baseZ = base['x'].to_numpy(),base['y'].to_numpy(),base['z'].to_numpy()

clip = min(len(methX),len(baseX))
methX,methY,methZ = meth['x'].to_numpy()[:clip],meth['y'].to_numpy()[:clip],meth['z'].to_numpy()[:clip]

baseX,baseY,baseZ = base['x'].to_numpy()[:clip],base['y'].to_numpy()[:clip],base['z'].to_numpy()[:clip]

dist =  np.sqrt(np.square(baseX-methX) + np.square(baseY-methY) + np.square(baseZ-methZ))

print(np.mean(dist))