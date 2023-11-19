import matplotlib.pyplot as plt


fname = '120000'
As, Bs, Cs, = [], [], []
X = []

with open(f"/Users/jeffreyburggraf/Desktop/Insure{fname}") as f:
    for line in f.readlines():
        x, a, b, c = map(lambda x: float(x.replace('$', '').replace(',','')), line.split())
        X.append(x)
        As.append(a)
        Bs.append(b)
        Cs.append(c)

for y, m, n in [(As,  'p', 'A'),  (Bs, 'o', 'B'), (Cs, '^', 'C')]:
    plt.plot(X, y, marker=m, label=f'{n} plan')

plt.title(f"${fname}")
plt.xlabel("Total medical bills")
plt.ylabel("(Premiums) + (out of pocket expenses)")
plt.legend()
plt.show()