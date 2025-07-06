import os
import tempfile
import gradio as gr
import pandas as pd
from transformers import pipeline

# Dictionary of available models
models = {
    "DistilBERT": "distilbert-base-uncased-finetuned-sst-2-english",
    "RoBERTa": "siebert/sentiment-roberta-large-english"
}

def classify_reviews(file, model_choice):
    """
    Reads a CSV file containing a 'review' column, runs sentiment analysis on each review using the selected model,
    and outputs a new CSV with the original reviews and their corresponding sentiment scores. 

    Args:
        file: CSV file
        model_choice (str): "DistilBERT" or "RoBERTa" to select a model.

    Returns:
        str: Path to the output CSV in the system's temp directory, or (None, error message) tuple if something goes wrong.
    """
    # Load CSV file into DataFrame
    try:
        df = pd.read_csv(file, encoding="cp1252", sep=";")
    except Exception as e:
        return None, f"Error loading CSV: {e}"
    
    # Validate presence of 'review' column
    if "review" not in df.columns:
        return None, "CSV must contain a 'review' column."
    
    # Check for missing values in the 'review' column
    if df["review"].isnull().any():
        return None, "Some reviews are missing (NaN)."
    
    # Ensure all reviews are non-empty strings
    if not all(isinstance(review, str) and review.strip() for review in df["review"]):
        return None, "The 'review' column must contain valid, non-empty text."
    
    # Initiate pipeline with the selected model
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model = models[model_choice],
            truncation = True,
            max_length = 512
        )
    except Exception as e:
        return None, f"Error loading model: {e}"

    # Converts reviews to a list of strings
    reviews = df["review"].tolist()

    # Run the sentiment analysis model
    try:
        results = classifier(reviews)
    except Exception as e:
        return None, f"Error during sentiment analysis {e}"
    
    # Add sentiment score to the DataFrame
    df["sentiment score"] = [result["score"] for result in results]
    
    # Output CSV and write to a temp file 
    output_df = df[["review", "sentiment score"]]

    file_name = f"sentiment_results_{model_choice}.csv"
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file_name)

    output_df.to_csv(temp_path, index=False)

    # Return the path to the generated CSV for download
    return temp_path, "Done! Download ready."

# Gradio interface for uploading and downloading CSV files
demo = gr.Interface(
    fn = classify_reviews,
    inputs = [
        gr.File(label="Upload CSV file with a 'review' column."),
        gr.Radio(["DistilBERT", "RoBERTa"], label="Select sentiment model", value="DistilBERT")
    ],
    outputs = [
        gr.File(label="Download results"),
        gr.Text(label="Status")
    ],
    title = "Sentiment classifier for reviews (CSV)",
    description = "Upload a CSV file with a 'review' column. Choose a model to classify each review. Download the output with sentiment scores.",
    flagging_mode="never"
)

# Launch the app on
if __name__ == "__main__":
    demo.launch()