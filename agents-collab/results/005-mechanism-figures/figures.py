"""Plot only saved checkpoints and graph-level metrics from development run 003."""
import argparse
import csv
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '003-bridge-selectivity'
FORCES = {'mean': 'Mean', 'huber': 'Fixed Huber', 'huber_local': 'Local Huber'}
DYNAMICS = ('wave', 'diffusion')
H = (0.5, 0.25)
GRAPH = 'two_block:sbm_24x24_bridges4_wb1_s0'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                     'svg.hashsalt': '003-development-figures', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#25364a', 'text.color': '#172b3e'})


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save(fig, out, name, rows):
    fig.savefig(out / f'{name}.svg', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)
    with (out / f'{name}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoints(out):
    values = []
    with np.load(SOURCE / 'part2_checkpoints.npz') as ck, np.load(SOURCE / 'part2_edges.npz') as ed:
        gi, = np.flatnonzero(ck['graph_ids'] == GRAPH)
        assert ed['graph_ids'][gi] == GRAPH
        edge = slice(*ed['edge_ptr'][gi:gi+2])
        i, j, bridge = (ed[key][edge] for key in ('edge_i', 'edge_j', 'edge_bridge'))
        times = ck['checkpoint_t']
        assert np.array_equal(times, (0, 50, 125, 250)) and bridge.sum() == 4
        gidx, forces, dynamics, h, states = (ck[k] for k in ('run_graph_index', 'run_force', 'run_dyn', 'run_h', 'states'))
        fig, ax = plt.subplots(2, 2, figsize=(8.4, 5.5), sharex=True, sharey=True)
        for panel, (force, dyn) in zip(ax.flat, product(('mean', 'huber_local'), DYNAMICS)):
            run, = np.flatnonzero((gidx == gi) & (forces == force) & (dynamics == dyn) & (h == 0.25))
            U = states[run]
            assert U.shape == (4, 8, 48) and np.isfinite(U).all()
            delta = U[..., j] - U[..., i]
            for label, group, color, marker in [('Bridge', bridge, '#bb4b2c', 'o'),
                                                 ('Internal', ~bridge, '#176e91', 's')]:
                rms = np.median(np.sqrt(np.mean(delta[..., group]**2, axis=-1)), axis=1)
                assert (rms > 0).all()
                panel.plot(times, rms, color=color, marker=marker, ls=':', lw=1.6, label=label)
                values.extend(dict(graph=GRAPH, force=force, dynamic=dyn, h=0.25, time=t,
                                   edge_class=label, n_starts=U.shape[1], n_edges=int(group.sum()), rms=y)
                              for t, y in zip(times, rms))
            panel.set_title(f'{FORCES[force]} · {dyn}', loc='left', weight='bold')
            panel.set_yscale('log')
            panel.set_xticks(times)
            panel.grid(axis='y', alpha=0.2)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.97, 0.88), ncols=2, frameon=False)
    fig.suptitle('Bridge and internal contrasts over time', x=0.10, ha='left', y=0.98, weight='bold', fontsize=17)
    fig.text(0.10, 0.90, 'One illustrative SBM · 24+24 nodes · 4 unit-weight bridges · seed 0 · h = 0.25')
    fig.supxlabel('Time (simulation units)', y=0.073)
    fig.supylabel('Edge-difference RMS (state units; log scale)', x=0.02)
    fig.text(0.10, 0.02, 'Within-class RMS, median over 8 starts. Markers are the only observed times.')
    fig.subplots_adjust(left=0.13, right=0.98, top=0.77, bottom=0.17, hspace=0.42, wspace=0.16)
    save(fig, out, 'checkpoint_contrasts', values)


def overview(out):
    saved = csv_rows(SOURCE / 'part2_scores.csv')
    gate = {(r['force'], r['dyn'], float(r['h'])): r for r in csv_rows(SOURCE / 'part2_gate.csv') if r['stratum'] == 'all'}
    scores = {}
    spectral = [r for r in saved if r['force'] == 'spectral']
    assert len({r['graph'] for r in spectral}) == len(spectral) == 48
    for r in saved:
        if r['window'] == 'late':
            key = r['force'], r['dyn'], float(r['h']), r['readout']
            group = scores.setdefault(key, {})
            assert r['graph'] not in group
            group[r['graph']] = r
    spectral_mean = np.mean([float(r['ari_plan']) for r in spectral])
    values = []
    fig, (fails, ari) = plt.subplots(1, 2, figsize=(9.8, 6.1), sharey=True,
                                    gridspec_kw={'width_ratios': (1, 2.8)})
    labels, ys = [], []
    for k, (force, dyn, h) in enumerate(product(FORCES, DYNAMICS, H)):
        n = int(gate[force, dyn, h]['graphs_any_failure'])
        assert int(gate[force, dyn, h]['graphs']) == 48
        y = k + 0.55*(k//4)
        ys.append(y)
        labels.append(f'{FORCES[force]} · {dyn} · {h:g}')
        fails.barh(y, n, color='#58758c', height=.45)
        fails.annotate(str(n), (n, y), xytext=(4, 0), textcoords='offset points', va='center')
        for j, (ro, color, label) in enumerate([('raw', '#7b60a1', 'Raw'), ('lp', '#148477', 'Low-pass')]):
            group = scores[force, dyn, h, ro]
            assert set(group) == {r['graph'] for r in spectral}
            assert sum(int(r['n_fail']) > 0 for r in group.values()) == n
            plans = np.array([float(r['ari_plan']) for r in group.values()])
            assert np.isfinite(plans).all()
            v = plans.mean()
            ari.barh(y + (j-.5)*.33, v, height=.31, color=color, label=label if k == 0 else None)
            values.append(dict(force=force, dynamic=dyn, h=h, readout=ro, n_graphs=48,
                               graphs_any_failure=n, mean_plan_ari=v))
    values.append(dict(force='spectral', dynamic='', h='', readout='reference', n_graphs=48,
                       graphs_any_failure='', mean_plan_ari=spectral_mean))
    fails.set(xlim=(0, 48), yticks=ys, yticklabels=labels, xlabel='Graphs with ≥1 failure (of 48)')
    fails.set_xticks([0, 12, 24, 36, 48])
    fails.invert_yaxis()
    ari.axvline(spectral_mean, color='#23374b', ls='--', lw=1.3, label=f'Spectral ({spectral_mean:.3f})')
    ari.set(xlabel='Mean late-window plan ARI (48 graphs)', xlim=(min(0, min(r['mean_plan_ari'] for r in values)), 1))
    ari.xaxis.set_major_locator(MultipleLocator(.2))
    for ax in (fails, ari):
        ax.set_axisbelow(True)
        ax.grid(axis='x', alpha=.2)
        ax.tick_params(axis='y', length=0)
    handles, labels = ari.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.97, 0.90), ncols=3, frameon=False)
    fig.suptitle('Development graph failures and late partition scores', x=0.06, ha='left', y=0.98,
                 weight='bold', fontsize=16)
    fig.text(0.06, 0.06, 'Rows show force · dynamic · step size. Eight starts per graph; any failure zeros plan ARI.')
    fig.text(0.06, 0.025, 'Late window: t = 125–250. Low-pass cutoff: 0.02 cycles per saved frame.')
    fig.subplots_adjust(left=0.27, right=0.98, top=0.81, bottom=0.15, wspace=0.22)
    save(fig, out, 'failure_and_ari', values)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints(args.out)
    overview(args.out)
