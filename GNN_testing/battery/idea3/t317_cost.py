"""T3.17 cost: CPU sec/step per force, n in {256,1024,4096} (sparse graphs: two cycle blocks / random_regular d=8). Single process."""
import i3
from i3 import C, np
import pandas as pd, time

FORCES = ["mean"] + i3.FORCES19 + ["owa_mc"]

def bench(g, f, R=8, steps=40, reps=3):
    u0 = C.init("uniform", g, R, np.random.default_rng(0))
    best = []
    for _ in range(reps):
        t0 = time.process_time(); C.run(g, "damp", f, u0, 0.5, steps, ps=dict(k=0.1), fail_amp=1e9); best.append((time.process_time() - t0) / steps)
    return float(np.min(best))

if __name__ == "__main__":
    rows = []
    for gk, mk in (("cycle2", lambda n: dict(kind="two_block", block="cycle", sizes=(n // 2, n // 2), bridges=2, seed=0)),
                   ("rr8", lambda n: dict(kind="random_regular", n=n, d=8, seed=0))):
        for n in (256, 1024, 4096):
            g, truth, meta = C.make_graph(mk(n))
            for f in FORCES:
                rows.append(dict(gkind=gk, n=g.n, dmax=int(g.dmax), force=f, R=8, sec_per_step=bench(g, f)))
            print(gk, n, "done", flush=True)
    C.write_csv(i3.RES / "t317_cost.csv", rows)
    df = pd.DataFrame(rows); pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    t = df.pivot_table(index="force", columns=["gkind", "n"], values="sec_per_step")
    t = t.div(t.loc["mean"], axis=1) if False else t
    print((t * 1000).round(2).to_string())
