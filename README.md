# Seasonly Sentiment Analysis on Vancouver Subreddit
This repository contains the code and files used for our course project in CMPT 353 at Simon Fraser University.

Our project's objective is to analyze the sentiments of Vancouver subreddit and its comments by applying statistical models and machine learning. We use statistical model, specifically the chi2_contingency from stats.scipy library to calculate the p-value in order to test our hypotheses.


# Requirements to run our program

- Git
- Python version 3.10 or higher
**libraries**:
    - sys
    - os 
    - Sparks version 3.5 or higher

To install library:
```
pip install --user scipy pandas numpy scikit-learn seaborn matplotlib datasets contractions
```

# Run the program

First we can clone this repository to our directory.

To run the sentiment analysis file in our terminal:

```
spark-submit <sentiment-analysis.py> output-2021-2023-submissions output-2021-2023-comments output
```

replace ```<sentiment-analysis.py>``` with any of the sentiment_analysis python file you wish to run.

To run the stats file:
```
spark-submit <reddit_stats.py> output-2021-2023-submissions output-2021-2023-comments stats-output-2021-2023
```

a stats-output-2021-2023 will appear in your directory with the plot images.

    





















