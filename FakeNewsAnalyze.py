'''import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from wordcloud import WordCloud
from collections import Counter


# Load dataset
pd.set_option('display.max_columns', None)
FakeNewsDataset = pd.read_csv("C:/Users/hello/Downloads/FakeNewsDetector.csv")
print(FakeNewsDataset.head())

# Handle missing text
FakeNewsDataset['text'] = FakeNewsDataset['text'].fillna('')

# Features = text column
X = FakeNewsDataset['text']
y = FakeNewsDataset['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)[:15]
plt.figure(figsize=(10, 6))
importances.plot(kind='barh')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.show()'''

'''import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load dataset
FakeNewsDataset = pd.read_csv("C:/Users/hello/Downloads/FakeNewsDetector.csv")

# Handle missing values
FakeNewsDataset['text'] = FakeNewsDataset['text'].fillna('')

# Function to clean/tokenize text
def get_words(texts):
    words = " ".join(texts).lower().split()
    words = [w for w in words if len(w) > 3 and w not in ENGLISH_STOP_WORDS]
    return words

# Subjects of interest
subjects = ["politicsNews", "worldnews", "News"]

# Loop through each subject
for subj in subjects:
    # Filter only FAKE news for this subject
    subset = FakeNewsDataset[
        (FakeNewsDataset['label'] == 1) & (FakeNewsDataset['subject'] == subj)
    ]

    words = get_words(subset['text'])
    counter = Counter(words).most_common(20)

    print(f"\n Top 20 words in FAKE news ({subj}):")
    for word, freq in counter:
        print(f"{word}: {freq}")

    # Bar plot
    plt.figure(figsize=(10,5))
    plt.bar([w for w, _ in counter], [c for _, c in counter], color="red")
    plt.title(f"Top 20 Words in Fake News - {subj}")
    plt.xticks(rotation=45)
    plt.show()

    # WordCloud
    wc = WordCloud(width=800, height=400, background_color="white", colormap="Reds").generate(" ".join(words))
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"WordCloud - Fake News ({subj})")
    plt.show()'''
import pandas as pd

# Load dataset
FakeNewsDataset = pd.read_csv("C:/Users/hello/Downloads/FakeNewsDetector.csv")

# Handle missing values
FakeNewsDataset['text'] = FakeNewsDataset['text'].fillna('')

# Define target subjects and keywords
subjects = ["politicsNews", "worldnews", "News"]
keywords = ["trump", "republican", "washington", "democratic", "campaign"]

# Filter: only fake news (label == 1) and subject in selected subjects
filtered_df = FakeNewsDataset[
    (FakeNewsDataset['label'] == 1) &
    (FakeNewsDataset['subject'].isin(subjects))
]

# Find rows containing any of the keywords
mask = filtered_df['title'].str.lower().apply(
    lambda x: any(kw in x for kw in keywords)
)

result_df = filtered_df[mask]

# Save matching texts to a txt file
with open("filtered_fake_news.txt", "w", encoding="utf-8") as f:
    for i, row in result_df.iterrows():
        f.write(f"--- Article ID: {i} | Title: {row['title']} ---\n")
        f.write(row['text'] + "\n\n")

print(f"Saved {len(result_df)} fake news articles containing {keywords} to filtered_fake_news.txt")
