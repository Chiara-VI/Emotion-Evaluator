# Emotion evaluator

This tool processes a CSV of text reviews, applies sentiment analysis using either a pretrained DistilBERT model (`distilbert-base-uncased-finetuned-sst-2-english`) or pretrained RoBERTa model (`siebert/sentiment-roberta-large-english`), and outputs a new CSV containing each review alongside its corresponding sentiment score.

## Table of contents

- [Project structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Features](#features)

## Project structure
```text
.
├── notebooks/
│   ├── benchmarks.ipynb               # Benchmark comparison between DistilBERT and RoBERTa
│   ├── data_exploration.ipynb         # Data exploration notebook
│   ├── distilBERT_sentiment.ipynb     # DistilBERT notebook, mainly used for benchmarks
│   ├── RoBERTa_sentiment.ipynb        # RoBERTa notebook, mainly used for benchmarks
│
├── scripts/
│   ├── sentiment_analysis.py          # Sentiment analysis script, can be used with either DistilBERT or RoBERTa
│   └── gradio_sentiment_demo.py       # Interactive demo of sentiment analysis
│
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```
## Dependencies

Install all requires packages:
```bash
pip install -r requirements.txt
```

## Usage
### Gradio Demo

Launch an interactive web app:
```bash
python scripts/gradio_sentiment_demo.py
```

### Sentiment analysis (Python script)

Process a CSV of reviews and save results:
```bash
python scripts/sentiment_analysis.py <input.csv> \ [--model distilbert|roberta] \ [--output-dir output_folder]
```

- `<input.csv>`: CSV file with reviews.
- `--model`: `distilbert` (default) or `roberta`.
- `--output-dir`: Folder for results (defaults to input file location).

**Output:** A new CSV in the output directory containing each original review and its corresponding sentiment score.

## Features

- Benchmark comparisons (DistilBERT vs. RoBERTa)  
- Data exploration notebook
- Flexible Python script with model choice  
- Interactive Gradio interface  
