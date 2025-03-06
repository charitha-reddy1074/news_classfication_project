import os
import pandas as pd
from transformers import pipeline

# List available datasets
datasets_path = os.path.join(os.getcwd(), 'datasets')
datasets = [f for f in os.listdir(datasets_path) if f.endswith('.csv')]

print("Available datasets:")
for i, dataset in enumerate(datasets):
    print(f"{i + 1}. {dataset}")

# Load and combine all datasets
dataframes = []
required_columns = {'news_headline', 'news_article', 'news_category'}
for dataset in datasets:
    dataset_path = os.path.join(datasets_path, dataset)
    data = pd.read_csv(dataset_path)
    if required_columns.issubset(data.columns):
        dataframes.append(data)
    else:
        print(f"Skipping {dataset} as it does not contain the required columns.")

if not dataframes:
    raise ValueError("No datasets with the required columns found.")

combined_data = pd.concat(dataframes, ignore_index=True)

# Preprocess data
def preprocess_data(data):
    data['text'] = data['news_headline'] + " " + data['news_article']
    data = data.rename(columns={'news_category': 'category'})
    return data[['text', 'category']]

processed_data = preprocess_data(combined_data)

# Define the label-to-category mapping based on the dataset
label_to_category = {
    "ARTS": "arts",
    "ARTS & CULTURE": "arts & culture",
    "BLACK VOICES": "black voices",
    "BUSINESS": "business",
    "COLLEGE": "college",
    "COMEDY": "comedy",
    "CRIME": "crime",
    "CULTURE & ARTS": "culture & arts",
    "DIVORCE": "divorce",
    "EDUCATION": "education",
    "ENTERTAINMENT": "entertainment",
    "ENVIRONMENT": "environment",
    "FIFTY": "fifty",
    "FOOD & DRINK": "food & drink",
    "GOOD NEWS": "good news",
    "GREEN": "green",
    "HEALTHY LIVING": "healthy living",
    "HOME & LIVING": "home & living",
    "IMPACT": "impact",
    "LATINO VOICES": "latino voices",
    "MEDIA": "media",
    "MONEY": "money",
    "PARENTING": "parenting",
    "PARENTS": "parents",
    "POLITICS": "politics",
    "QUEER VOICES": "queer voices",
    "RELIGION": "religion",
    "SCIENCE": "science",
    "SPORTS": "sports",
    "STYLE": "style",
    "STYLE & BEAUTY": "style & beauty",
    "TASTE": "taste",
    "TECH": "technology",
    "THE WORLDPOST": "the worldpost",
    "TRAVEL": "travel",
    "U.S. NEWS": "u.s. news",
    "WEDDINGS": "weddings",
    "WEIRD NEWS": "weird news",
    "WELLNESS": "wellness",
    "WOMEN": "women",
    "WORLD NEWS": "world news",
    "WORLDPOST": "worldpost"
}


# Load the pre-trained model for text classification
classifier = pipeline("text-classification", model="dima806/news-category-classifier-distilbert")

def classify_news(article):
    max_length = 512
    truncated_article = article[:max_length]
    result = classifier(truncated_article)
    label = result[0]['label']

    # Ensure the returned label exists in the mapping
    category = label_to_category.get(label, label)  # If not found, return the same categorY
    if category is None:
        raise ValueError(f"Unexpected label '{label}' returned by the model. Expected one of {list(label_to_category.keys())}")

    return category


# Example usage to classify a sample article
#sample_article = "The latest Hollywood superhero film has shattered box office records, earning over $500 million in its opening weekend. Critics praise its stunning visual effects and compelling storyline, making it a must-watch for movie lovers."
#category = classify_news(sample_article)
#print(f"The news belongs to {category} category!")