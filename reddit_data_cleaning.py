import sys
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('reddit extracter').config("spark.ui.showConsoleProgress", "true").getOrCreate()

reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/'
reddit_comments_path = '/courses/datasets/reddit_comments_repartitioned/'
output = 'reddit-subset-2021-2023'

comments_schema = types.StructType([
    types.StructField('body', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

submissions_schema = types.StructType([
    types.StructField('selftext', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])

def main(inputs):
    
    reddit_submissions_data = spark.read.json(inputs + '/submissions', schema=submissions_schema)
    reddit_comments_data = spark.read.json(inputs + '/comments', schema=comments_schema)
    
    remove_text = ['', '[deleted', '[removed]']
    
    reddit_submissions_data = reddit_submissions_data.where(~functions.col('selftext').isin(remove_text))
    reddit_comments_data = reddit_comments_data.where(~functions.col('body').isin(remove_text))
    
    reddit_submissions_data.write.json(inputs + '/submissions', mode='overwrite', compression='gzip')
    reddit_comments_data.write.json(inputs + '/comments', mode='overwrite', compression='gzip')
    
    
inputs = sys.argv[1]
assert spark.version >= '3.5' # make sure we have Spark 3.5+
spark.sparkContext.setLogLevel('WARN')

main(inputs)