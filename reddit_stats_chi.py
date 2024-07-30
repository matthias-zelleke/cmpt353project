import sys
assert sys.version_info >= (3, 10)  # make sure we have Python 3.10+
from pyspark.sql import SparkSession, functions, types
import pandas as pd
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
    types.StructField('sentiment', types.LongType()),
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
    types.StructField('sentiment', types.LongType()),
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

    # Initialize SparkSession
    spark = SparkSession.builder.appName('example code').getOrCreate()
    spark.sparkContext.setLogLevel('WARN')

    # Read and filter data
    reddit_submissions_data = spark.read.json(input_submissions, schema=submissions_schema)
    reddit_comments_data = spark.read.json(input_comments, schema=comments_schema)

    summer_months = [6, 7, 8, 9]
    winter_months = [1, 2, 3, 4, 5, 10, 11, 12]
    

    # Chi-squared Test

    summer_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    winter_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(winter_months))

    summer_submissions_neg = summer_submissions_filter.where(functions.col('sentiment') == 0)
    summer_submissions_pos = summer_submissions_filter.where(functions.col('sentiment') == 4)
    
    winter_submissions_neg = winter_submissions_filter.where(functions.col('sentiment') == 0)
    winter_submissions_pos = winter_submissions_filter.where(functions.col('sentiment') == 4)
    
    summer_sub_neg_counts = summer_submissions_neg.count()
    summer_sub_pos_counts = summer_submissions_pos.count()
    
    winter_sub_neg_counts = winter_submissions_neg.count()
    winter_sub_pos_counts = winter_submissions_pos.count()
    
    print(summer_sub_pos_counts, summer_sub_neg_counts, winter_sub_pos_counts, winter_sub_neg_counts)
    contingency_submissions = [[summer_sub_pos_counts, summer_sub_neg_counts],
                   [winter_sub_pos_counts, winter_sub_neg_counts]]
    
    chi2_submissions = chi2_contingency(contingency_submissions)
    
    print(f'Chi-squared p-value submissions: ', chi2_submissions.pvalue)
    


    summer_comments_filter = reddit_comments_data.where(functions.col('month').isin(summer_months))
    winter_comments_filter = reddit_comments_data.where(functions.col('month').isin(winter_months))

    summer_comments_neg = summer_comments_filter.where(functions.col('sentiment') == 0)
    summer_comments_pos = summer_comments_filter.where(functions.col('sentiment') == 4)

    winter_comments_neg = winter_comments_filter.where(functions.col('sentiment') == 0)
    winter_comments_pos = winter_comments_filter.where(functions.col('sentiment') == 4)

    summer_com_neg_counts = summer_comments_neg.count()
    summer_com_pos_counts = summer_comments_pos.count()

    winter_com_neg_counts = winter_comments_neg.count()
    winter_com_pos_counts = winter_comments_pos.count()

    print(summer_com_pos_counts, summer_com_neg_counts, winter_com_pos_counts, winter_com_neg_counts)
    contingency_comments = [[summer_com_pos_counts, summer_com_neg_counts],
               [winter_com_pos_counts, winter_com_neg_counts]]

    chi2_comments = chi2_contingency(contingency_comments)
    
    print(f'Chi-squared p-value comments: ', chi2_comments.pvalue)

    
if __name__ == "__main__":
    input_submissions = sys.argv[1]
    input_comments = sys.argv[2]
    output = sys.argv[3]
    main(input_submissions, input_comments, output)






#spark-submit reddit_stats_chi.py reddit-subset-2021/submissions reddit-subset-2021/comments stats_outputs