import pandas as pd
import re

print("📥 Loading original CSV...")
df = pd.read_csv("keyword_edge_final.csv")
print(f"➡ Loaded shape: {df.shape}")

def clean(text):
    text = str(text)
    text = re.sub(r"[.,!?]", "", text)
    text = text.strip()
    return text

print("🧹 Cleaning...")
df["source"] = df["source"].map(clean)
df["target"] = df["target"].map(clean)

df = df.dropna(subset=["source", "target"])
df = df[df["source"] != ""]
df = df[df["target"] != ""]

print(f"✅ Cleaned shape: {df.shape}")
print("💾 Saving...")
df.to_csv("keyword_edge_final_cleaned.csv", index=False)
print("🎉 Done.")
