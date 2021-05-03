# from JSB_tools.nuke_data_tools import Nuclide
from matplotlib import pyplot as plt
import matplotlib; print(matplotlib.__version__)
print(matplotlib.get_backend())

from matplotlib import pyplot as plt
plt.errorbar(x=[1, 2], y=[1, 2], yerr=[1, 2], ds='steps-mid')
# plt.errorbar(x=[1, 2], y=[1,2], yerr=[1,2], ds='steps-mid')

# plt.show()