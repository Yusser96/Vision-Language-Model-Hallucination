#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect likely hallucinations in LLM outputs by comparing them to a reference text.

For each (reference, llm_output) pair, this script:
- builds a "support" vocabulary from the reference using spaCy lemmas
- flags LLM tokens that are NOUN/PROPN/ADJ whose lemmas are unseen in the support
- flags noun chunks whose content lemmas are mostly unseen (>= threshold)
- writes a CSV report and prints a brief summary

Notes
------
- The default spaCy model is "de_core_news_sm" to mirror your original code.
  Use "-m en_core_web_sm" for English, or any pipeline that includes a parser.
- Input files are line-aligned; if lengths differ, shorter files are padded with empty lines.

Examples
--------
$ python detect_hallucinations.py \
    -r outs_spamo/output_shak_original/refs.txt \
    -l outs_spamo/output_shak_original/preds.txt \
    -o llm_based_output_shak_original.csv \
    -m de_core_news_sm
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Set, Tuple
import json

import pandas as pd
import spacy

# Content POS to check in LLM output
CONTENT_POS: Set[str] = {"NOUN", "PROPN", "ADJ"}


def load_lines(p: Path, encoding: str = "utf-8") -> List[str]:
    with p.open("r", encoding=encoding) as f:
        return [line.strip() for line in f.readlines()]


def normalize_for_set(doc) -> Set[str]:
    """
    Return a set of lowercased lemmas for content-like tokens from a spaCy Doc.

    We include a broader set than CONTENT_POS to reduce false positives when
    paraphrasing (e.g., verb vs noun forms).
    """
    allowed = {"NOUN", "PROPN", "VERB", "AUX", "ADJ", "ADV", "NUM"}
    return {
        tok.lemma_.lower()
        for tok in doc
        if (tok.is_alpha or tok.pos_ == "NUM") and tok.pos_ in allowed
    }


def hallucinated_tokens(doc_llm, support_lemmas: Set[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Find LLM tokens that are NOUN/PROPN/ADJ whose lemma isn't in the support set.
    Returns (surface_forms, lemmas).
    """
    flagged: List[str] = []
    flagged_lemmas: List[str] = []
    all_tokens: List[str] = []
    for tok in doc_llm:
        if tok.pos_ in CONTENT_POS and (tok.is_alpha or tok.pos_ == "NUM"):
            lemma = tok.lemma_.lower()
            if lemma not in support_lemmas:
                flagged.append(tok.text)
                flagged_lemmas.append(lemma)
            all_tokens.append(lemma)
    return flagged, flagged_lemmas, all_tokens


def hallucinated_noun_chunks(doc_llm, support_lemmas: Set[str], threshold: float) -> List[str]:
    """
    Find noun chunks in the LLM output whose constituent content lemmas are mostly unseen.
    Flags a chunk if (unseen content lemmas / all content lemmas in chunk) >= threshold.
    """
    chunks_flagged: List[str] = []
    for chunk in doc_llm.noun_chunks:
        content_lemmas = [
            t.lemma_.lower()
            for t in chunk
            if t.pos_ in CONTENT_POS and (t.is_alpha or t.pos_ == "NUM")
        ]
        if not content_lemmas:
            continue
        unseen = [l for l in content_lemmas if l not in support_lemmas]
        if len(unseen) / len(content_lemmas) >= threshold:
            chunks_flagged.append(chunk.text)
    return chunks_flagged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="detect_hallucinations.py",
        description="Detect likely hallucinations in line-aligned LLM outputs using spaCy.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-r", "--reference",
        type=Path,
        required=True,
        help="Path to reference lines file (one sample per line).",
    )
    parser.add_argument(
        "-l", "--llm",
        type=Path,
        required=True,
        help="Path to LLM output lines file (one sample per line).",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./hallucination_report.csv"),
        help="Output CSV path (default: ./hallucination_report.csv).",
    )
    parser.add_argument(
        "-o-jsonl", "--output-jsonl",
        type=Path,
        default=Path("./hallucination_report.jsonl"),
        help="Output CSV path (default: ./hallucination_report.csv).",
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="de_core_news_sm",
        help="spaCy model to load (default: de_core_news_sm).",
    )
    parser.add_argument(
        "--chunk-threshold",
        type=float,
        default=0.6,
        help="Threshold for flagging noun chunks (fraction unseen, default: 0.6).",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8",
        help="File encoding for input text files (default: utf-8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load spaCy model and ensure a parser is present (needed for noun_chunks)
    try:
        nlp = spacy.load(args.model, disable=["ner", "textcat"])
    except Exception as e:
        raise SystemExit(
            f"Could not load spaCy model '{args.model}'. "
            f"Install with e.g. `python -m spacy download {args.model}`.\nOriginal error: {e}"
        )
    if not nlp.has_pipe("parser"):
        raise SystemExit(
            "The loaded spaCy model must include a parser to extract noun_chunks."
        )

    # Read inputs
    ref_lines = load_lines(args.reference, args.encoding)
    llm_lines = load_lines(args.llm, args.encoding)

    # Line alignment + padding
    n = max(len(ref_lines), len(llm_lines))
    if len(ref_lines) < n:
        ref_lines += [""] * (n - len(ref_lines))
    if len(llm_lines) < n:
        llm_lines += [""] * (n - len(llm_lines))

    rows = []
    for i in range(n):
        ref = ref_lines[i]
        llm_text = llm_lines[i]

        # Build support vocabulary from reference only
        doc_ref = nlp(ref)
        support = normalize_for_set(doc_ref)

        doc_llm = nlp(llm_text)

        flagged_tokens, flagged_lemmas, content_tokens = hallucinated_tokens(doc_llm, support)

        _, _, ref_content_tokens = hallucinated_tokens(doc_ref, support)
        flagged_chunks = hallucinated_noun_chunks(doc_llm, support, args.chunk_threshold)

        rows.append(
            {
                "sample_id": i,
                "reference": ref,
                "llm_output": llm_text,
                "hallucinated_tokens": ", ".join(flagged_tokens),
                "hallucinated_lemmas": ", ".join(sorted(set(flagged_lemmas))),
                "hallucinated_noun_chunks": ", ".join(sorted(set(flagged_chunks))),
                "num_flagged_tokens": len(flagged_tokens),
                "num_flagged_chunks": len(set(flagged_chunks)),
                "llm_len_tokens": sum(1 for t in doc_llm if not t.is_space),
                "support_vocab_size": len(support),
                "content_tokens":content_tokens,
                "support":support,
                "flagged_lemmas":flagged_lemmas,
                "ref_content_tokens":ref_content_tokens
            }
        )

    out_path = Path(args.output_jsonl) if args.output_jsonl else None
    out_f = out_path.open("w", encoding="utf-8") if out_path else None
    if out_f:
        for row in rows:
            #print(rec)
            Sentence_CHAIR_I = 0
            if row["content_tokens"]:
                Sentence_CHAIR_I = len(row["flagged_lemmas"])/len(row["content_tokens"])
            rec = {"idx": row["sample_id"]+1, "pred": row["llm_output"].strip(), "refs": [row["reference"].strip()], "pred_objs": list(row["content_tokens"]), "ref_objs": list(row["ref_content_tokens"]), "hallucinated": list(row["flagged_lemmas"]), "Sentence_CHAIR_I": Sentence_CHAIR_I}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.close()

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False, quoting=csv.QUOTE_MINIMAL)

    # Summary
    total_samples = len(rows)
    any_flagged = sum(
        1 for r in rows if r["num_flagged_tokens"] > 0 or r["num_flagged_chunks"] > 0
    )
    print(f"Processed {total_samples} samples.")
    print(
        f"Samples with suspected hallucinations: {any_flagged} "
        f"({(any_flagged / total_samples * 100.0) if total_samples else 0:.1f}%)"
    )
    print(f"Report written to: {args.output}")


if __name__ == "__main__":
    main()
