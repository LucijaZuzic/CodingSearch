import pandas as pd 

df = pd.read_csv("table_of_papers.csv", encoding = "UTF-8")
set_ref = set()
dys = dict()
dy = dict()
for i in range(len(df["Reference"])):
	x = df["Reference"][i]
	if x not in set_ref:
		y = df["Year"][i]
		if str(y) == 'nan':
			start = 0
			for k in range(len(x) - 3):
				if x[k] == '2' and x[k + 1] == '0' and x[k + 2] >= '0' and x[k + 2] <= '1' and x[k + 3] >= '0' and x[k + 3] <= '9':
					start = k
			y = int(x[start:start + 4])
		if y not in dy:
			dys[y] = set()
			dy[y] = []
		dys[y].add(x)
		dy[y].append(x)
	set_ref.add(x) 
print(len(set_ref))
for y in sorted(list(dy.keys())):
	print(y, len(dy[y]), len(dys[y]))
