import pandas as pd
df = pd.read_csv("iris.data", header=None)
print(df[0:10].to_latex())
print(df[-10:].to_latex())
