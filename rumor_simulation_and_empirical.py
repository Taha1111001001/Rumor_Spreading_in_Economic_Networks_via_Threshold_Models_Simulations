
# rumor_simulation_and_empirical.py
# This script runs synthetic threshold simulations (ER, BA, WS) and performs empirical validation
# on a CSV of adjusted close prices. It produces figures saved to an output directory.
# Requirements: numpy, pandas, networkx, matplotlib, yfinance (optional)
# Usage: python rumor_simulation_and_empirical.py --prices market_data.csv

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random

def run_threshold_sim(G, theta, initial_seed_frac=0.01, max_steps=200):
    nodes = list(G.nodes())
    N = len(nodes)
    if N == 0:
        return [0], 0.0
    adopted = set(random.sample(nodes, max(1,int(initial_seed_frac*N))))
    ts = [len(adopted)/N]
    for step in range(max_steps):
        new = set()
        for v in nodes:
            if v in adopted:
                continue
            neigh = list(G.neighbors(v))
            if len(neigh)==0:
                continue
            adopted_neigh = sum(1 for u in neigh if u in adopted)
            if adopted_neigh / len(neigh) >= theta:
                new.add(v)
        if not new:
            break
        adopted |= new
        ts.append(len(adopted)/N)
    return ts, len(adopted)/N

def synthetic_experiments(outdir):
    N = 2000
    avg_k = 4
    p = avg_k / (N-1)
    # ER
    G_er = nx.erdos_renyi_graph(N, p, seed=42)
    # BA
    m = 2
    G_ba = nx.barabasi_albert_graph(N, m, seed=42)
    # WS
    k = 4
    G_ws = nx.watts_strogatz_graph(N, k, 0.1, seed=42)

    thetas = np.linspace(0.05, 0.6, 12)
    results = {}
    for name, G in [('ER', G_er), ('BA', G_ba), ('WS', G_ws)]:
        means, stds = [], []
        for theta in thetas:
            vals = []
            for _ in range(8):
                _, f = run_threshold_sim(G, theta, initial_seed_frac=0.01)
                vals.append(f)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        results[name] = (thetas, np.array(means), np.array(stds))

    # Plot
    plt.figure(figsize=(7,4))
    for name in results:
        thetas, means, stds = results[name]
        plt.errorbar(thetas, means, yerr=stds, marker='o', label=name, capsize=3)
    plt.xlabel('Threshold theta')
    plt.ylabel('Final adoption fraction')
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'synthetic_final_vs_theta.png'), dpi=200)
    plt.close()

def empirical_experiments(prices_csv, outdir):
    prices = pd.read_csv(prices_csv, index_col=0)
    returns = np.log(prices.astype(float)).diff().dropna()
    corr = returns.corr()
    corr.to_csv(os.path.join(outdir, 'empirical_correlation.csv'))

    # build network by thresholding corr
    corr_threshold = 0.5
    G = nx.Graph()
    tickers = corr.columns.tolist()
    G.add_nodes_from(tickers)
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if j <= i: continue
            if corr.loc[ti, tj] >= corr_threshold:
                G.add_edge(ti, tj, weight=float(corr.loc[ti, tj]))

    # Save correlation heatmap
    plt.figure(figsize=(6,5))
    plt.imshow(corr.values, vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(tickers)), tickers, rotation=90, fontsize=6)
    plt.yticks(range(len(tickers)), tickers, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'correlation_matrix.png'), dpi=200)
    plt.close()

    # Save network visualization
    plt.figure(figsize=(6,5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=200)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'market_network.png'), dpi=200)
    plt.close()

    # run contagion sweep
    thetas = np.linspace(0.05, 0.6, 12)
    means, stds = [], []
    for theta in thetas:
        vals = []
        for _ in range(20):
            _, f = run_threshold_sim(G, theta, initial_seed_frac=0.03)
            vals.append(f)
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    pd.DataFrame({'theta':thetas, 'mean_final':means, 'std_final':stds}).to_csv(os.path.join(outdir, 'empirical_summary.csv'), index=False)

    # plot results
    plt.figure(figsize=(7,4))
    plt.errorbar(thetas, means, yerr=stds, marker='o', capsize=3)
    plt.xlabel('Threshold theta')
    plt.ylabel('Final adoption fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'empirical_final_vs_theta.png'), dpi=200)
    plt.close()

    # example timeseries
    ts, _ = run_threshold_sim(G, 0.15, initial_seed_frac=0.03)
    plt.figure(figsize=(6,4))
    plt.plot(range(len(ts)), ts, marker='o')
    plt.xlabel('Time step')
    plt.ylabel('Adopted fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'empirical_time_series.png'), dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices', type=str, default='market_data.csv', help='CSV of adjusted close prices (index=date, columns=tickers)')
    parser.add_argument('--outdir', type=str, default='outputs', help='output directory for figures and csvs')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    synthetic_experiments(args.outdir)
    if os.path.exists(args.prices):
        empirical_experiments(args.prices, args.outdir)
    else:
        print('Prices CSV not found at', args.prices)

if __name__ == '__main__':
    main()

