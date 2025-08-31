#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze hallucination report CSVs produced by detect_hallucinations.py and
generate summary tables, charts, and an HTML report.

What it does
------------
- Parses list-like CSV columns (hallucinated_*).
- Computes per-sample metrics (e.g., hallucinated token rate, any_hallucination).
- Aggregates frequency tables for lemmas/tokens/noun chunks.
- Saves sorted per-sample CSVs and top-N frequency CSVs.
- Produces PNG charts (histograms, scatter plots).
- Builds a self-contained HTML overview.

Inputs
------
A CSV created by your detector, with at least these columns:
    sample_id, reference, llm_output,
    hallucinated_tokens, hallucinated_lemmas, hallucinated_noun_chunks,
    num_flagged_tokens, num_flagged_chunks, llm_len_tokens, support_vocab_size

Examples
--------
$ python analyze_hallucinations.py \
    -i llm_based_output_shak_original.csv \
    -o ./report

$ python analyze_hallucinations.py \
    --input results.csv --outdir ./out --top-k 100 --bins 40 --title "My Run"

Outputs
-------
<outdir>/
  ├─ report.html
  ├─ samples_sorted_by_rate.csv
  ├─ samples_with_hallucinations.csv
  ├─ samples_without_hallucinations.csv
  ├─ top_hallucinated_lemmas.csv
  ├─ top_hallucinated_tokens.csv
  ├─ top_hallucinated_chunks.csv
  ├─ hist_hallucinated_token_rate.png
  ├─ hist_flagged_token_count.png
  ├─ hist_llm_len.png
  ├─ scatter_len_vs_rate.png
  └─ scatter_support_vs_rate.png
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Template
from collections import Counter


EXPECTED_COLUMNS = {
    "sample_id", "reference", "llm_output",
    "hallucinated_tokens", "hallucinated_lemmas", "hallucinated_noun_chunks",
    "num_flagged_tokens", "num_flagged_chunks", "llm_len_tokens", "support_vocab_size",
}


# ---------- Helpers ----------

def _split_list_field(s: str) -> List[str]:
    """Split a comma-separated list field from the CSV into a list of items.
    Handles empty strings/NaN gracefully.
    """
    if pd.isna(s) or str(s).strip() == "":
        return []
    # The detector joined with ", " but we defensively split on ","
    return [x.strip() for x in str(s).split(",") if x.strip()]


def safe_div(a: float, b: float) -> float:
    return (a / b) if b else 0.0


# ---------- Main analysis ----------

def analyze(df: pd.DataFrame, outdir: Path, *, top_k: int = 50, bins: int = 30, title: str = "Hallucination Analysis Report") -> Dict[str, float]:
    os.makedirs(outdir, exist_ok=True)

    # --- Parse list-like columns into lists ---
    df["hallucinated_lemmas_list"] = df["hallucinated_lemmas"].apply(_split_list_field)
    df["hallucinated_tokens_list"] = df["hallucinated_tokens"].apply(_split_list_field)
    df["hallucinated_noun_chunks_list"] = df["hallucinated_noun_chunks"].apply(_split_list_field)

    # --- Derived metrics ---
    df["halluc_token_rate"] = df.apply(lambda r: safe_div(r["num_flagged_tokens"], r["llm_len_tokens"]), axis=1)
    df["any_hallucination"] = (df["num_flagged_tokens"] > 0) | (df["num_flagged_chunks"] > 0)

    # --- Summary stats ---
    n = len(df)
    n_any = int(df["any_hallucination"].sum())
    pct_any = 100 * safe_div(n_any, n)

    avg_len = float(df["llm_len_tokens"].mean())
    avg_flagged = float(df["num_flagged_tokens"].mean())
    avg_rate = float(df["halluc_token_rate"].mean())

    # Correlations (length, support size vs hallucination rate)
    corr_len_rate = float(df["llm_len_tokens"].corr(df["halluc_token_rate"]))
    corr_support_rate = float(df["support_vocab_size"].corr(df["halluc_token_rate"]))

    # --- Frequency tables ---
    lemma_counter = Counter()
    token_counter = Counter()
    chunk_counter = Counter()
    for lemmas in df["hallucinated_lemmas_list"]:
        lemma_counter.update(lemmas)
    for toks in df["hallucinated_tokens_list"]:
        token_counter.update(toks)
    for ch in df["hallucinated_noun_chunks_list"]:
        chunk_counter.update(ch)

    top_lemmas = pd.DataFrame(lemma_counter.most_common(top_k), columns=["lemma", "count"])
    top_tokens = pd.DataFrame(token_counter.most_common(top_k), columns=["token", "count"])
    top_chunks = pd.DataFrame(chunk_counter.most_common(top_k), columns=["noun_chunk", "count"])

    # Save frequency tables
    top_lemmas.to_csv(os.path.join(outdir, "top_hallucinated_lemmas.csv"), index=False)
    top_tokens.to_csv(os.path.join(outdir, "top_hallucinated_tokens.csv"), index=False)
    top_chunks.to_csv(os.path.join(outdir, "top_hallucinated_chunks.csv"), index=False)

    # --- Per-sample exports (sorted views) ---
    df.sort_values(["halluc_token_rate", "num_flagged_tokens"], ascending=False)\
      .to_csv(os.path.join(outdir, "samples_sorted_by_rate.csv"), index=False)

    df[df["any_hallucination"]].to_csv(os.path.join(outdir, "samples_with_hallucinations.csv"), index=False)
    df[~df["any_hallucination"]].to_csv(os.path.join(outdir, "samples_without_hallucinations.csv"), index=False)

    # --- Visualizations ---
    # Histogram: hallucinated token rate
    plt.figure()
    df["halluc_token_rate"].hist(bins=bins)
    plt.xlabel("Hallucinated token rate (flagged tokens / LLM tokens)")
    plt.ylabel("Number of samples")
    plt.title("Distribution of hallucinated token rate")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_hallucinated_token_rate.png"), dpi=160)
    plt.close()

    # Histogram: flagged token counts
    plt.figure()
    df["num_flagged_tokens"].hist(bins=bins)
    plt.xlabel("Number of flagged tokens per sample")
    plt.ylabel("Number of samples")
    plt.title("Distribution of flagged token counts")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_flagged_token_count.png"), dpi=160)
    plt.close()

    # Histogram: LLM output length
    plt.figure()
    df["llm_len_tokens"].hist(bins=bins)
    plt.xlabel("LLM output length (tokens)")
    plt.ylabel("Number of samples")
    plt.title("Distribution of LLM output length")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "hist_llm_len.png"), dpi=160)
    plt.close()

    # Scatter: LLM length vs hallucination rate
    plt.figure()
    plt.scatter(df["llm_len_tokens"], df["halluc_token_rate"], alpha=0.6)
    plt.xlabel("LLM output length (tokens)")
    plt.ylabel("Hallucinated token rate")
    plt.title(f"Length vs Hallucination Rate (corr={corr_len_rate:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_len_vs_rate.png"), dpi=160)
    plt.close()

    # Scatter: support vocab size vs rate
    plt.figure()
    plt.scatter(df["support_vocab_size"], df["halluc_token_rate"], alpha=0.6)
    plt.xlabel("Support vocab size (reference)")
    plt.ylabel("Hallucinated token rate")
    plt.title(f"Support Size vs Hallucination Rate (corr={corr_support_rate:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "scatter_support_vs_rate.png"), dpi=160)
    plt.close()

    # Bar: Top lemmas
    if not top_lemmas.empty:
        plt.figure(figsize=(10, 6))
        plt.bar(top_lemmas["lemma"].astype(str), top_lemmas["count"].astype(int))
        plt.xticks(rotation=70, ha="right")
        plt.xlabel("Hallucinated lemmas")
        plt.ylabel("Count across samples")
        plt.title("Top hallucinated lemmas")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "bar_top_lemmas.png"), dpi=160)
        plt.close()

    # --- Build HTML report ---
    html_template = Template("""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{{ title }}</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px;}
    h1, h2, h3 { margin-top: 1.2em; }
    .kpi { display:flex; gap:24px; flex-wrap:wrap; }
    .card { border:1px solid #e4e4e7; border-radius:10px; padding:16px; min-width:240px; }
    table { border-collapse: collapse; margin-top: 12px; }
    th, td { border:1px solid #e4e4e7; padding:6px 10px; }
    img { max-width: 900px; width:100%; height:auto; border:1px solid #eee; border-radius: 8px; }
    code { background:#f6f7f8; padding:2px 4px; border-radius:4px;}
    a { text-decoration:none; color:#0b66c3; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>

  <div class="kpi">
    <div class="card"><b>Total samples</b><div>{{ n }}</div></div>
    <div class="card"><b>Samples with hallucination(s)</b><div>{{ n_any }} ({{ pct_any | round(1) }}%)</div></div>
    <div class="card"><b>Avg LLM length (tokens)</b><div>{{ avg_len | round(2) }}</div></div>
    <div class="card"><b>Avg flagged tokens</b><div>{{ avg_flagged | round(2) }}</div></div>
    <div class="card"><b>Avg hallucinated token rate</b><div>{{ avg_rate | round(3) }}</div></div>
  </div>

  <h2>Distributions</h2>
  <img src="hist_hallucinated_token_rate.png" alt="hist rate"/>
  <img src="hist_flagged_token_count.png" alt="hist flagged"/>
  <img src="hist_llm_len.png" alt="hist length"/>

  <h2>Correlations</h2>
  <p>Length vs hallucination rate (corr={{ corr_len_rate | round(2) }})</p>
  <img src="scatter_len_vs_rate.png" alt="scatter len vs rate"/>
  <p>Support vocab size vs hallucination rate (corr={{ corr_support_rate | round(2) }})</p>
  <img src="scatter_support_vs_rate.png" alt="scatter support vs rate"/>

  <h2>Top Hallucinated Lemmas</h2>
  {% if has_top_lemmas %}
    <img src="bar_top_lemmas.png" alt="bar top lemmas"/>
    <p><a href="top_hallucinated_lemmas.csv">Download CSV</a></p>
  {% else %}
    <p>No hallucinated lemmas found.</p>
  {% endif %}

  <h2>Downloads</h2>
  <ul>
    <li><a href="samples_sorted_by_rate.csv">Samples sorted by hallucination rate</a></li>
    <li><a href="samples_with_hallucinations.csv">Samples with hallucinations</a></li>
    <li><a href="samples_without_hallucinations.csv">Samples without hallucinations</a></li>
    <li><a href="top_hallucinated_tokens.csv">Top hallucinated tokens</a></li>
    <li><a href="top_hallucinated_chunks.csv">Top hallucinated noun chunks</a></li>
  </ul>

  <h2>How to read this</h2>
  <ul>
    <li><b>Hallucinated token rate</b> = flagged content tokens ÷ total LLM tokens per sample.</li>
    <li><b>Top lemmas</b> reveal frequently introduced concepts not supported by the reference.</li>
    <li>A negative correlation with <i>support vocab size</i> suggests richer reference coverage reduces hallucinations.</li>
  </ul>

  <p style="margin-top:40px;color:#666">Generated automatically by analyze_hallucinations.py</p>
</body>
</html>
""")

    html = html_template.render(
        title=title,
        n=n,
        n_any=n_any,
        pct_any=pct_any,
        avg_len=avg_len,
        avg_flagged=avg_flagged,
        avg_rate=avg_rate,
        corr_len_rate=0.0 if np.isnan(corr_len_rate) else corr_len_rate,
        corr_support_rate=0.0 if np.isnan(corr_support_rate) else corr_support_rate,
        has_top_lemmas=not top_lemmas.empty,
    )
    with open(os.path.join(outdir, "report.html") , "w", encoding="utf-8") as f:
        f.write(html)

    print("=== Summary ===")
    print(f"Total samples: {n}")
    print(f"Samples with hallucinations: {n_any} ({pct_any:.1f}%)")
    print(f"Avg LLM length (tokens): {avg_len:.2f}")
    print(f"Avg flagged tokens: {avg_flagged:.2f}")
    print(f"Avg hallucinated token rate: {avg_rate:.3f}")
    print(f"Correlation (length vs rate): {0.0 if np.isnan(corr_len_rate) else corr_len_rate:.2f}")
    print(f"Correlation (support size vs rate): {0.0 if np.isnan(corr_support_rate) else corr_support_rate:.2f}")
    print(f"\nArtifacts written to: {outdir}")
    print(f"Open: {os.path.join(outdir, 'report.html')}")

    return {
        "n": n,
        "n_any": n_any,
        "pct_any": pct_any,
        "avg_len": avg_len,
        "avg_flagged": avg_flagged,
        "avg_rate": avg_rate,
        "corr_len_rate": 0.0 if np.isnan(corr_len_rate) else corr_len_rate,
        "corr_support_rate": 0.0 if np.isnan(corr_support_rate) else corr_support_rate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="analyze_hallucinations.py",
        description="Analyze a hallucination report CSV and generate charts + an HTML summary.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the detector's CSV output.",
    )
    parser.add_argument(
        "-o", "--outdir",
        type=Path,
        default=Path("./report"),
        help="Directory to write artifacts (default: ./report).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="How many top lemmas/tokens/chunks to include (default: 50).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=30,
        help="Histogram bin count (default: 30).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Hallucination Analysis Report",
        help="HTML report title (default: 'Hallucination Analysis Report').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load CSV
    df = pd.read_csv(args.input)
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing expected columns: {sorted(missing)}")

    analyze(
        df=df,
        outdir=args.outdir,
        top_k=args.top_k,
        bins=args.bins,
        title=args.title,
    )


if __name__ == "__main__":
    main()
