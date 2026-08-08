# Current Milestone 3 Results

## Attached dataset audit

- Raw rows: **25,777**
- Missing narratives: **0**
- Rows removed for fewer than 10 normalized words: **94**
- Exact normalized duplicates removed: **107**
- Rows retained for discovery: **25,576**
- Older-consumer share: **9.8%**
- Median narrative length: **162 words**
- Regex scam-audit match rate: **65.8%**

## Corpus-filtering decision

The keyword filter is not used to remove records. Manual inspection found clear false negatives,
including advance-fee winnings, fake rental listings, and marketplace/payment-release scams.
The attached file also closely matches the approximately 25,000-record working corpus stated in the
Milestone 2 proposal. The final approach therefore keeps the complete cleaned corpus and relies on
HDBSCAN noise handling plus manual topic review.

## Original BERTopic baseline from the teammate notebook

- Five non-noise topics
- 2,922 noise complaints
- Largest topic: 19,812 complaints
- DBCV: 0.040
- Mean C_v coherence: 0.447

This baseline is retained for comparison but is not suitable as the final label set because the
largest topic dominates the corpus and several topic keywords are stopwords.

## Additional preliminary baseline

A TF-IDF + MiniBatchKMeans development baseline was tested on a fixed 3,000-row sample:

- k = 8
- cosine silhouette: 0.012
- largest cluster share: 29.2%
- one collapsed one-record cluster

The low silhouette and collapsed cluster show that lexical KMeans is not a strong replacement for
semantic BERTopic. It remains a documented baseline only.

## Next required run

Run `notebooks/01_dataset_preparation_revised.ipynb` in an environment that can download
`sentence-transformers/all-MiniLM-L6-v2`. It will cache embeddings, compare HDBSCAN settings,
calculate DBCV and C_v coherence, and create `topic_review_template.csv`.
