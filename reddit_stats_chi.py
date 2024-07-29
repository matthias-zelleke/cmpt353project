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

    # Filter and collect scores for submissions
    summer_submissions = reddit_submissions_data.where(functions.col('month').isin(summer_months)).select('score').rdd.flatMap(lambda x: x).collect()
    winter_submissions = reddit_submissions_data.where(functions.col('month').isin(winter_months)).select('score').rdd.flatMap(lambda x: x).collect()

    # Filter and collect scores for comments
    summer_comments = reddit_comments_data.where(functions.col('month').isin(summer_months)).select('score').rdd.flatMap(lambda x: x).collect()
    winter_comments = reddit_comments_data.where(functions.col('month').isin(winter_months)).select('score').rdd.flatMap(lambda x: x).collect()

    # Mann-Whitney U Test for submissions
    stat_subs, p_value_subs = mannwhitneyu(summer_submissions, winter_submissions, alternative='less')

    # Mann-Whitney U Test for comments
    stat_comms, p_value_comms = mannwhitneyu(summer_comments, winter_comments, alternative='less')

    print(f'Mann-Whitney U test statistic subs: {stat_subs}')
    print(f'P-value subs: {p_value_subs}')
    print(f'Mann-Whitney U test statistic comments: {stat_comms}')
    print(f'P-value comments: {p_value_comms}')

    # Chi-squared Test

    summer_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    winter_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(winter_months))

    summer_submissions_neg = summer_submissions_filter.where(functions.col('sentiment') == 0)
    summer_submissions_pos = summer_submissions_filter.where(functions.col('sentiment') == 4)
    
    winter_submissions_neg = winter_submissions_filter.where(functions.col('sentiment') == 0)
    winter_submissions_pos = winter_submissions_filter.where(functions.col('sentiment') == 4)
    
    summer_neg_counts = summer_submissions_neg.count()
    summer_pos_counts = summer_submissions_pos.count()
    
    winter_neg_counts = winter_submissions_neg.count()
    winter_pos_counts = winter_submissions_pos.count()
    
    print(summer_pos_counts, summer_neg_counts, winter_pos_counts, winter_neg_counts)
    contingency = [[summer_pos_counts, summer_neg_counts],
                   [winter_pos_counts, winter_neg_counts]]
    
    #summer_counts = summer_submissions_filter.groupBy('subreddit').agg(functions.sum('num_comments').alias('count')).collect()
    #winter_counts = winter_submissions_filter.groupBy('subreddit').agg(functions.sum('num_comments').alias('count')).collect()

    #summer_dict = {row['subreddit']: row['count'] for row in summer_counts}
    #winter_dict = {row['subreddit']: row['count'] for row in winter_counts}

    #all_subreddits = set(summer_dict.keys()).union(set(winter_dict.keys()))

    chi2 = chi2_contingency(contingency)
    
    print(f'P-value submissions: ', chi2.pvalue)

    '''
    summer_counts = summer_submissions_filter.groupBy('subreddit').agg(functions.count('comments').alias('count')).collect()
    winter_counts = winter_submissions_filter.groupBy('subreddit').agg(functions.count('comments').alias('count')).collect()

    # Convert to dictionaries for easy manipulation
    summer_dict = {row['subreddit']: row['count'] for row in summer_counts}
    winter_dict = {row['subreddit']: row['count'] for row in winter_counts}

    # Get a set of all unique subreddits
    all_subreddits = set(summer_dict.keys()).union(set(winter_dict.keys()))

    # Construct the contingency table
    contingency = [
        [summer_dict.get(subreddit, 0) for subreddit in all_subreddits],
        [winter_dict.get(subreddit, 0) for subreddit in all_subreddits]
    ]

    print("Contingency Table:")
    for row in contingency:
        print(row)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    print(f'Chi-squared test statistic: {chi2}')
    print(f'P-value: {p_value}')
    print(f'Degrees of freedom: {dof}')
    print('Expected frequencies:')
    for row in expected:
        print(row)

    '''

    #contingency = [[],[]]
    
    #summer_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    #summer_submissions_comments_count = summer_submissions_filter.groupby('subreddit').agg(functions.count('comments')).rdd.flatMap(lambda x: x).collect()[0]
    #contingency[0].append(summer_submissions_comments_count)

    #winter_submissions_filter = reddit_submissions_data.where(functions.col('month').isin(winter_months))
    #winter_submissions_comments_count =  winter_submissions_filter.groupby('subreddit').agg(functions.count('comments')).rdd.flatMap(lambda x: x).collect()[0]
    #contingency[1].append(winter_submissions_comments_count)
    
    #chi2_subs = chi2_contingency(contingency)
    #print(contingency, chi2_subs)
    

    #print(f'Chi-squared test statistic subs: {chi2_subs}')
    #print(f'P-value subs: ', chi2_subs.pvalue)


    
if __name__ == "__main__":
    input_submissions = sys.argv[1]
    input_comments = sys.argv[2]
    output = sys.argv[3]
    main(input_submissions, input_comments, output)






#spark-submit reddit_stats_chi.py reddit-subset-2021/submissions reddit-subset-2021/comments stats_outputs