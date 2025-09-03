# Vision-Language-Model-Hallucination

## automatic diff
Find **content words** that appear in predictions but **not** in references.

- `pip install spacy pandas`
- `python3 -m spacy download de_core_news_sm`
- `python3 detect_hallucinations.py -r "outs_spamo/output_shak_original/refs.txt" -l "outs_spamo/output_shak_original/preds.txt" -o "./llm_based_output_shak_original.csv"`
- `python3 analyze_hallucinations.py -i "./llm_based_output_shak_original.csv" -o "./report2"`

Results: `report2/report.html`

## CHAIR
**How it works.** CHAIR (Caption Hallucination Assessment with Image Relevance) measures object hallucination by comparing **objects mentioned in predictions** to **objects licensed by the references**. It reports two rates: **CHAIR-S** (captions with ≥1 hallucinated object / all captions) and **CHAIR-I** (hallucinated objects / all mentioned objects). In this repo, “objects” are our **glosses** from `gloss.txt`; you can expand coverage with `synonyms_semantic.tsv`.

- `pip install spacy pandas`
- `python3 -m spacy download de_core_news_md`
  
- `python3 gchair.py output_shak_original/preds.txt output_shak_original/refs.txt --inventory gloss.txt`
- `python3 gchair.py output_shak_original/preds.txt output_shak_original/refs.txt --inventory gloss.txt --synonyms synonyms_semantic.tsv`

- `python3 gchair.py output_shak_original/preds.txt output_shak_original/refs.txt \
 --inventory gloss.txt  \
 --out results_gchair_lemm.jsonl \
  --synonyms synonyms_semantic.tsv \
  --spacy-model de_core_news_md`

**Reference**  
Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., & Saenko, K. (2018). *Object Hallucination in Image Captioning*. EMNLP.
