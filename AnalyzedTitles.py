import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

FakeNewsDataset = pd.read_csv("C:/Users/hello/Downloads/FakeNewsDetector.csv")

FakeNewsDataset['text'] = FakeNewsDataset['text'].fillna('')
FakeNewsDataset['title'] = FakeNewsDataset['title'].fillna('')

subjects = ["politicsNews", "worldnews", "News"]
keywords = ["trump", "said", "u.s.", "president", "republican"]

filtered_df = FakeNewsDataset[
    (FakeNewsDataset['label'] == 1) &
    (FakeNewsDataset['subject'].isin(subjects))
]

mask = filtered_df['title'].str.lower().apply(
    lambda x: any(kw in x for kw in keywords)
)
result_df = filtered_df[mask]

vectorizer = CountVectorizer(ngram_range=(2,3), stop_words="english")
X = vectorizer.fit_transform(result_df['title'].str.lower())
counts = X.sum(axis=0).A1
vocab = vectorizer.get_feature_names_out()

ngram_freq = Counter(dict(zip(vocab, counts))).most_common(15)
top_patterns = [ngram for ngram, _ in ngram_freq]

print("\nTop 15 patterns:")
for ngram, freq in ngram_freq:
    print(f"{ngram} → {freq}")

title_matches = []
for title in result_df['title']:
    lower_title = title.lower()
    matched = [pattern for pattern in top_patterns if pattern in lower_title]
    if matched:
        title_matches.append((title, len(matched), matched))

title_matches_sorted = sorted(title_matches, key=lambda x: x[1], reverse=True)

with open("template_matching_titles.txt", "w", encoding="utf-8") as f:
    f.write("Titles with biggest matching templates:\n\n")
    for title, count, matched in title_matches_sorted[:20]:  # топ-20
        f.write(f"[{count} templates] {title}\n")
        f.write(f"   Matching templates: {', '.join(matched)}\n\n")

print(f"\nSaved {len(title_matches_sorted)} title in template_matching_titles.txt")
print("\nTop titles:")
print(f"{title_matches_sorted[0][0]} → {title_matches_sorted[0][1]} templates")