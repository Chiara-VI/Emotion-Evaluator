import os
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

models = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta": "siebert/sentiment-roberta-large-english"
}

def analyze_sentiment(input_path, model_name, output_dir):
    """
    Perform sentiment analysis on reviews from a CSV file and save the results.

    Args:
        input_path (str): Path to the input CSV file containing a 'review' column.
        model_name (str): Name of the sentiment model to use. Default = 'distilbert'.
        output_dir (str): Directory where the output CSV with sentiment scores will be saved.

    Returns:
        Saves results to a CSV file. 
    """

   # Validate input file
    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    # Validate model
    if model_name not in models:
        print(f"Error: Model '{model_name}' not supported. Choose 'distilbert' or 'roberta'.")
        return
    
    # Validate output directory
    if not os.path.isdir(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return

    # Load dataset
    df = pd.read_csv(input_path, encoding="cp1252", sep=";")

    # Check if the 'review' column exist in the DataFrame
    if "review" not in df.columns:
        print("Error: 'review' column not found in input CSV.")
        return
    
    # Check if there are any missing (NaN) values in the 'review' column
    if df["review"].isnull().any():
        print("Error: Some reviews are missing (NaN values) in the 'review' column.")
        return

    # Initiate sentiment analysis pipeline
    print(f"Using model: {model_name}")
    try:
        classifier = pipeline(
            "sentiment-analysis",
            model = models[model_name],
            truncation = True,
            max_length = 512
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Analyze sentiment
    reviews = df["review"].tolist()
    results = list(tqdm(classifier(reviews), desc="Processing"))

    # Add sentiment score predictions to DataFrame
    df["sentiment score"] = [result["score"] for result in results]
    
    # Save results
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    score_output = os.path.join(output_dir, f"{model_name}_{file_name}_sentiment_scores.csv")
    df[["review", "sentiment score"]].to_csv(score_output, index=False)

    print(f"Sentiment analysis complete. Saved to: {score_output}")

def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis on reviews.")

    # Required positional argument: input file
    parser.add_argument(
        "input_file",
        help="Path to input CSV file containing reviews."
    )

    # Optional: choose model    
    parser.add_argument(
        "--model",
        choices=["distilbert", "roberta"],
        default="distilbert",
        help="Model to use: 'distilbert' (default) or 'roberta'."
    )

    # Optional: output directory (must already exist)
    parser.add_argument(
        "--output-dir",
        help="Directory to save results. If not set, saves to the same folder as the input file."
    )

    args = parser.parse_args()

    # Default output dir = same as input file's folder
    output_dir = args.output_dir or os.path.dirname(args.input_file) or "."

    analyze_sentiment(args.input_file, args.model, output_dir)

if __name__ == "__main__":
    main()