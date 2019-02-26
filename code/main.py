import pandas as pd
import numpy as np

a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

df = pd.DataFrame({"name1" : a, "name2" : b})
df.to_csv("submission2.csv", index=True)