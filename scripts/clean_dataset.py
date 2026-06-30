"""
Script tiền xử lý dataset.
Clean caption text, loại bỏ duplicates, lọc low quality pairs.
"""
import re
import pandas as pd


def clean_caption(text):
    text = str(text).lower()
    # Remove product codes (words containing numbers)
    text = re.sub(r"\b[a-z0-9]*\d+[a-z0-9]*\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    gender = ""
    for g in ["men's", "women's", "men", "women", "unisex", "boys", "girls", "kids", "girl's", "boy's"]:
        if f" {g} " in f" {text} " or text.startswith(f"{g} ") or text.endswith(f" {g}"):
            gender = g
            text = re.sub(rf"\b{g}\b", "", text).strip()

    text = re.sub(r"\s+", " ", text).strip()

    if gender:
        clean_text = f"a {text} for {gender}."
    else:
        clean_text = f"a {text}."

    clean_text = (
        clean_text.replace("a a ", "a ")
        .replace("a e", "an e")
        .replace("a i", "an i")
        .replace("a o", "an o")
        .replace("a u", "an u")
    )
    clean_text = clean_text.capitalize()

    # Expand caption based on product category
    t = clean_text.lower()
    if "shirt" in t or "top" in t or "polo" in t:
        clean_text += " It features a comfortable neckline and sleeves. The material is soft and provides a relaxed fit."
    elif "pant" in t or "jean" in t or "legging" in t or "short" in t or "trouser" in t:
        clean_text += " It is made of a stretchy material and features a comfortable waistband. It is designed to fit snugly."
    elif "shoe" in t or "sneaker" in t or "boot" in t or "sandal" in t:
        clean_text += " It features a durable sole and a comfortable fit for everyday wear."
    elif "dress" in t or "skirt" in t or "saree" in t:
        clean_text += " It has a beautiful design and is made of a comfortable, flowy material."
    elif "bag" in t or "backpack" in t or "purse" in t:
        clean_text += " It has multiple compartments with secure closures and a stylish design."
    elif "jacket" in t or "coat" in t or "hoodie" in t:
        clean_text += " It has a front closure and pockets. The material is designed to be very comfortable."
    elif "watch" in t:
        clean_text += " It features a classic dial and a durable, comfortable strap."
    else:
        clean_text += " It is made of high-quality materials and features a stylish, modern design."

    return clean_text


def clean_data(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    original_len = len(df)

    # Apply NLP caption cleaning
    df["caption"] = df["caption"].apply(clean_caption)

    # Sort by similarity_score (descending)
    if "similarity_score" in df.columns:
        df = df.sort_values("similarity_score", ascending=False)

    # Drop duplicates
    df = df.drop_duplicates(subset=["caption"], keep="first")
    df = df.drop_duplicates(subset=["image_path"], keep="first")
    unique_len = len(df)

    # Filter low quality pairs
    if "similarity_score" in df.columns:
        df = df[df["similarity_score"] >= 0.30]

    final_len = len(df)

    df.to_csv(output_csv, index=False)

    print(f"Original rows: {original_len}")
    print(f"After removing duplicates: {unique_len}")
    print(f"After filtering low quality (score < 0.3): {final_len}")
    print(f"Clean dataset saved to: {output_csv}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        clean_data(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python clean_dataset.py <input.csv> <output.csv>")
