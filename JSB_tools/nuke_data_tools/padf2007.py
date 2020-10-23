from matplotlib import pyplot as plt
import openmc

e = openmc.data.endf.Evaluation("/Users/jeffreyburggraf/PycharmProjects/PHELIX/Xs/proton_activation/18040")

r = openmc.data.Reaction.from_endf(e, 5)

for prod in r.products:
    print(type(prod.particle))
    if str(prod.particle) == "Cl39":
        plt.plot(prod.yield_.x / 1E6, prod.yield_.y * 1E3)
        break
    print(prod.particle, prod.yield_.y)
    print(dir(prod))

    #

plt.show()
print(r.products)
print(r)
