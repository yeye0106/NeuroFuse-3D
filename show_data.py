import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
df = pd.read_csv("label.csv")
print(df.head())
print(df.columns)
print(df.shape)



sub = list(df['Subject'])
gr = list(df['Group'])
print()
print(f"Subject去重：{len(set(sub))}")
print()
print("分组样本如下：\nAD:%d, CN:%d, MCI:%d"%(gr.count("AD"), gr.count("CN"), gr.count("MCI")))
print()

de = {}
for i in list(df['Description']):
    if i in de:
        de[i] += 1
    else:
        de[i] = 1

print("Description如下：")
max_key_len = max(len(key) for key in de)
for i in de:
    print(f"{i:<{max_key_len}} : {de[i]}")