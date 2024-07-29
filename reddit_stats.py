import sys
assert sys.version_info >= (3, 10) # make sure we have Python 3.10+
from pyspark.sql import SparkSession, functions, types
import pandas as pd
from datasets import load_dataset
from scipy.stats import mannwhitneyu, chi2_contingency

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

submissions_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('created', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('domain', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.BooleanType()),
    types.StructField('from', types.StringType()),
    types.StructField('from_id', types.StringType()),
    types.StructField('from_kind', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('hide_score', types.BooleanType()),
    types.StructField('id', types.StringType()),
    types.StructField('is_self', types.BooleanType()),
    types.StructField('link_flair_css_class', types.StringType()),
    types.StructField('link_flair_text', types.StringType()),
    types.StructField('media', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('over_18', types.BooleanType()),
    types.StructField('permalink', types.StringType()),
    types.StructField('quarantine', types.BooleanType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('saved', types.BooleanType()),
    types.StructField('score', types.LongType()),
    types.StructField('secure_media', types.StringType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('stickied', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('thumbnail', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('url', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

def main(input_submissions, input_comments, output):

    # Read and filter data
    reddit_submissions_data = spark.read.json(input_submissions, schema=submissions_schema)
    reddit_comments_data = spark.read.json(input_comments, schema=comments_schema)

    summer_months = [6, 7, 8, 9]
    winter_months = [1, 2, 3, 4, 5, 10, 11, 12]

    summer_submissions = reddit_submissions_data.where(functions.col('month').isin(summer_months)).select('score').rdd.flatMap(lambda x: x).collect()
    winter_submissions = reddit_submissions_data.where(functions.col('month').isin(winter_months)).select('score').rdd.flatMap(lambda x: x).collect()

    summer_comments = reddit_comments_data.where(functions.col('month').isin(summer_months)).select('score').rdd.flatMap(lambda x: x).collect()
    winter_comments = reddit_comments_data.where(functions.col('month').isin(winter_months)).select('score').rdd.flatMap(lambda x: x).collect()

    # Print number of data points for debugging
    print(f"Number of summer submissions: {len(summer_submissions)}")
    print(f"Number of winter submissions: {len(winter_submissions)}")
    print(f"Number of summer comments: {len(summer_comments)}")
    print(f"Number of winter comments: {len(winter_comments)}")

    # Mann-Whitney U Test
    stat_subs, p_value_subs = mannwhitneyu(summer_submissions, winter_submissions, alternative='less')
    stat_comms, p_value_comms = mannwhitneyu(summer_comments, winter_comments, alternative='less')

    print("Summer Submissions Scores:", summer_submissions[:10])  # Print first 10 scores for brevity
    print("Winter Submissions Scores:", winter_submissions[:10])  # Print first 10 scores for brevity
    print("Summer Comments Scores:", summer_comments[:10])  # Print first 10 scores for brevity
    print("Winter Comments Scores:", winter_comments[:10])  # Print first 10 scores for brevity

    print(f'Mann-Whitney U test statistic subs: {stat_subs}')
    print(f'P-value subs: {p_value_subs}')
    print(f'Mann-Whitney U test statistic comments: {stat_comms}')
    print(f'P-value comments: {p_value_comms}')

    results = {
        'mannwhitneyu_statistic subs': stat_subs,
        'mannwhitneyu_p_value subs': p_value_subs,
        'mannwhitneyu_statistic comments': stat_comms,
        'mannwhitneyu_p_value comments': p_value_comms
    }

    # Save results
    # pd.DataFrame([results]).to_csv(f'{output}/mannwhitneyu_results.csv', index=False)

input_submissions = sys.argv[1]
input_comments = sys.argv[2]
output = sys.argv[3]
spark = SparkSession.builder.appName('example code').getOrCreate()
#assert spark.version >= '3.5' # make sure we have Spark 3.5+
spark.sparkContext.setLogLevel('WARN')
#sc = spark.sparkContext

main(input_submissions, input_comments, output)


#spark-submit reddit_stats.py reddit-subset-2021/submissions reddit-subset-2021/comments stats_output