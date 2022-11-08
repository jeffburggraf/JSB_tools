import time
from openmc.data import Evaluation, Reaction
from multiprocessing import Process, Queue
from pathlib import Path



def _load(p_, q):
    t0 = time.time()
    reactions = []

    try:
        e = Evaluation(p_)
        for _, mt, _, _ in e.reaction_list:
            r = Reaction.from_endf(e, mt)
            reactions.append(r.products)

    except (KeyError, ValueError):
        e = None

    print(f"{p_.name}. took {time.time() - t0} seconds")

    # q.put(1)
    if q is not None:
        q.put(reactions)
    else:
        return reactions


def read_all_evaluations(path, imax=None, paralell=False):
    t0 = time.time()
    Q = Queue()

    procs = []

    outs = []

    for i, path in enumerate(iter_paths(path)):
        if imax is not None and i > imax:
            print("Reached max files")
            break
        if paralell:
            p = Process(target=_load, args=(path, Q))
            p.start()
            procs.append(p)

            if len(procs) == 10:
                [p.join() for p in procs]
                outs.extend([Q.get() for _ in range(len(procs))])
                procs = []
        else:
            outs.append(_load(path, None))

    print(f"TOTAL TIME = {time.time() - t0:.1f}")
    return outs


if __name__ == '__main__':
    pass
