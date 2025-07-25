import pandas as pd, re, string, nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

def clean(txt):
    txt = re.sub(r"http\S+|www\S+", "", txt.lower())
    txt = re.sub(f"[{re.escape(string.punctuation)}]", " ", txt)
    return " ".join(w for w in txt.split()
                    if w not in stopwords.words("english"))

fake = pd.read_csv("data/raw/Fake.csv"); fake["label"] = 1
real = pd.read_csv("data/raw/True.csv"); real["label"] = 0
df = pd.concat([fake, real])[["title", "text", "label"]]
df["clean"] = (df["title"] + " " + df["text"]).apply(clean)
df.to_parquet("data/processed/news.parquet", index=False)
print("✅ Cleaned file saved → data/processed/news.parquet")