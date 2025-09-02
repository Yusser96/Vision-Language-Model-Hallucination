#!/usr/bin/env python3
# chairs_gloss.py — CHAIR for sign-language captions using your own gloss inventory
# Usage:
#   python chairs_gloss.py preds.txt refs.txt --inventory gloss_inventory.txt [--synonyms synonyms.tsv] [--count-repeats]
#
# File formats:
#   - preds.txt: one predicted caption per line (gloss tokens separated by spaces)
#   - refs.txt:  one reference per line; if multiple references for the same item, join with " || "
#   - gloss_inventory.txt: one canonical gloss per line (e.g., HOUSE, GIVE-TO, PERSON)
#   - synonyms.tsv (optional): TSV with two columns: <alias>\t<canonical>, where <canonical> is in the inventory
#
# Notes:
#   - By default, objects are *unique types per caption* (set semantics). Add --count-repeats to count repeated mentions.
#   - If --inventory is omitted, the inventory is inferred from tokens observed in references.

import argparse, re, sys
from collections import Counter

def norm_text(s: str) -> str:
    # Uppercase, collapse whitespace; keep letters, digits, hyphen, underscore, plus.
    s = s.upper()
    s = re.sub(r"[^A-Z0-9_\-\+\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_refs(line: str) -> list:
    return [part.strip() for part in line.split("||")]

def tokenize(s: str) -> list:
    return norm_text(s).split()

def load_inventory(path: str, refs_lines=None) -> set:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            inv = {norm_text(line).strip() for line in f if line.strip()}
        return inv
    # Infer from references (all tokens seen in refs)
    inv = set()
    for line in refs_lines:
        for ref in split_refs(line):
            inv.update(tokenize(ref))
    return inv

def load_synonyms(path: str, inventory: set) -> dict:
    if not path:
        return {}
    alias2canon = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            if "\t" not in raw:
                print(f"[warn] synonyms.tsv line {i} has no TAB; skipping: {raw}", file=sys.stderr)
                continue
            alias, canon = raw.split("\t", 1)
            alias = norm_text(alias)
            canon = norm_text(canon)
            if canon not in inventory:
                print(f"[warn] canonical gloss '{canon}' not in inventory; skipping alias '{alias}'", file=sys.stderr)
                continue
            alias2canon[alias] = canon
    return alias2canon

def extract_objects(text: str, inventory: set, alias2canon: dict, count_repeats: bool):
    """Return mentioned objects as a set (default) or Counter (if count_repeats)."""
    toks = tokenize(text)
    mapped = []
    for t in toks:
        if t in alias2canon:
            mapped.append(alias2canon[t])
        elif t in inventory:
            mapped.append(t)
        # else: token not considered an object
    if count_repeats:
        return Counter(mapped)
    return set(mapped)

def union_of_refs(ref_line: str, inventory: set, alias2canon: dict, count_repeats: bool):
    """Combine objects across multiple reference captions (OR across refs)."""
    objs = Counter() if count_repeats else set()
    for ref in split_refs(ref_line):
        r = extract_objects(ref, inventory, alias2canon, count_repeats)
        if count_repeats:
            # treat refs as support: we only need to know if an object can appear;
            # so convert to set of types before union
            for k in r.keys():
                objs[k] = 1  # presence flag
        else:
            objs |= r
    return objs

def chair_metrics(preds, refs, inventory, alias2canon, count_repeats=False):
    if len(preds) != len(refs):
        raise ValueError(f"Line count mismatch: {len(preds)} preds vs {len(refs)} refs")

    captions_total = len(preds)
    captions_with_halluc = 0
    mentioned_total = 0
    hallucinated_total = 0
    examples = []

    for i, (p, rline) in enumerate(zip(preds, refs)):
        pred_objs = extract_objects(p, inventory, alias2canon, count_repeats)
        ref_objs  = union_of_refs(rline, inventory, alias2canon, count_repeats)

        if count_repeats:
            # Hallucinated = items where predicted count > 0 but ref count == 0
            halluc_types = {k for k, v in pred_objs.items() if ref_objs.get(k, 0) == 0}
            halluc_count = sum(pred_objs[k] for k in halluc_types)
            mentioned_count = sum(pred_objs.values())
        else:
            halluc_types = pred_objs - ref_objs
            halluc_count = len(halluc_types)
            mentioned_count = len(pred_objs)

        if halluc_types:
            captions_with_halluc += 1
            if len(examples) < 5:
                examples.append((i+1, p.strip(), rline.strip(), sorted(halluc_types)))

        hallucinated_total += halluc_count
        mentioned_total += mentioned_count

    chair_s = (captions_with_halluc / captions_total) if captions_total else 0.0
    chair_i = (hallucinated_total / mentioned_total) if mentioned_total else 0.0

    return chair_s, chair_i, {
        "captions_total": captions_total,
        "captions_with_halluc": captions_with_halluc,
        "mentioned_total": mentioned_total,
        "hallucinated_total": hallucinated_total,
        "examples": examples,
    }

def main():
    ap = argparse.ArgumentParser(description="CHAIR for sign-language glosses with custom inventory.")
    ap.add_argument("preds", help="predictions file (one caption per line)")
    ap.add_argument("refs", help="references file (one or more refs per line; join multiple with ' || ')")
    ap.add_argument("--inventory", "-i", help="path to gloss inventory (one canonical gloss per line). If omitted, inferred from references.", default=None)
    ap.add_argument("--synonyms", "-s", help="optional TSV mapping alias<TAB>canonical (canonical must be in inventory).", default=None)
    ap.add_argument("--count-repeats", action="store_true", help="count repeated mentions for CHAIR-I (multiset); default is unique types.")
    args = ap.parse_args()

    with open(args.preds, "r", encoding="utf-8") as f:
        preds = [line.rstrip("\n") for line in f]
    with open(args.refs, "r", encoding="utf-8") as f:
        refs = [line.rstrip("\n") for line in f]

    inventory = load_inventory(args.inventory, refs_lines=refs)
    alias2canon = load_synonyms(args.synonyms, inventory)

    chair_s, chair_i, info = chair_metrics(preds, refs, inventory, alias2canon, count_repeats=args.count_repeats)

    print(f"Inventory size: {len(inventory)}  |  Synonyms: {len(alias2canon)}")
    print(f"CHAIR_S (sentence-level): {chair_s:.4f}  = {info['captions_with_halluc']} / {info['captions_total']}")
    print(f"CHAIR_I (instance-level): {chair_i:.4f}  = {info['hallucinated_total']} / {info['mentioned_total']}")

    if info["examples"]:
        print("\nExamples (up to 5):")
        for idx, cap, r, objs in info["examples"]:
            print(f"  pred {idx}: {cap}")
            print(f"  ref  {idx}: {r}")
            print(f"    → hallucinated: {', '.join(objs)}")

if __name__ == "__main__":
    main()
