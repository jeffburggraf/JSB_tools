from pathlib import Path


p1 = Path(__file__)
p2 = Path(__file__)

print(p1 == p2)