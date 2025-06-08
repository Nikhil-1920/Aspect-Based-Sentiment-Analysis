import nltk
import os

nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required resources
nltk.download('punkt', download_dir=nltk_data_dir)
nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
nltk.download('sentiwordnet', download_dir=nltk_data_dir)

print(f"NLTK resources downloaded to {nltk_data_dir}")
print(f"NLTK data path is now: {nltk.data.path}")