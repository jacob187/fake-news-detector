import re


# Function to clean text
def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>", "", text)
    text = text.replace("\n", " ")
    return text
