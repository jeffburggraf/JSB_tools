import numpy as np

def write_CSV(ergs_list, xss_list):
    ERGS = []
    XSS = []

    def add_columns(ergs: list, xss: list):
        if len(ERGS) == 0:
            pass
        else:
            size = len(ERGS[0])
            if len(ergs) < size:
                pad = size - len(ergs)
                ergs.extend([np.nan] * pad)
                xss.extend([np.nan] * pad)

            if len(ergs) > size:
                pad = len(ergs) - size
                for i in range(len(ERGS)):
                    ERGS[i].extend([np.nan] * pad)
                    XSS[i].extend([np.nan] * pad)

        ERGS.append(ergs)
        XSS.append(xss)

    for i in range(len(ergs_list)):
        add_columns(ergs_list[i], xss_list[i])

    ergs_xss = []
    for ergs, xss in zip(ERGS, XSS):
        ergs_xss.extend([ergs, xss])

    rows = np.array(ergs_xss).transpose()

    def to_string(x):
        if np.isnan(x):
            return ""
        else:
            return str(x)

    with open("xs.csv", 'w') as f:
        for row in rows:
            line = (", ".join(map(to_string, row)) + '\n')
            f.write(line)


exalpe_ergs = [[-10, -5, -2, -1],
        [2, 4, 6],
        [-10, -20]]
example_xss = [[1, 2, 3, 4],
       [1,2,3],
       [40, 100]]

write_CSV(exalpe_ergs, example_xss)
