# Vision-Language-Model-Hallucination

- pip install spacy pandas
- python3 -m spacy download de_core_news_sm

- python3 detect_hallucinations.py -r "outs_spamo/output_shak_original/refs.txt" -l "outs_spamo/output_shak_original/preds.txt" -o "./llm_based_output_shak_original.csv"
- !python3 analyze_hallucinations.py -i "./llm_based_output_shak_original2.csv" -o "./report2"

