import sys
assert sys.version_info >= (3, 10) # make sure we have Python 3.10+
from pyspark.sql import SparkSession, functions, types
import pandas as pd
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader_analyzer = SentimentIntensityAnalyzer()

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


def get_sentiment_vader(text):
    return vader_analyzer.polarity_scores(text)

def main(input_submissions, input_comments, output):
    
    reddit_submissions_data = spark.read.json(input_submissions, schema=submissions_schema)
    reddit_comments_data = spark.read.json(input_comments, schema=comments_schema)
    
    
    ### Analysis #3, Treating June-August as summer months:
    
    summer_months = [6, 7, 8] # the months of the year where Vancouver has sunny weather
    winter_months = [1, 2, 3, 4, 5, 9, 10, 11, 12]
    
    reddit_submissions_summer = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    reddit_submissions_summer = reddit_submissions_summer.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions_winter = reddit_submissions_data.where(functions.col('month').isin(winter_months))
    reddit_submissions_winter = reddit_submissions_winter.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_summer = reddit_comments_data.where(functions.col('month').isin(summer_months))
    reddit_comments_summer = reddit_comments_summer.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_winter = reddit_comments_data.where(functions.col('month').isin(winter_months))
    reddit_comments_winter = reddit_comments_winter.withColumn('id', functions.monotonically_increasing_id())
    
    
    training_data = load_dataset('sentiment140', trust_remote_code=True)['train']
    training_data = training_data.to_pandas()
    training_data = training_data.sample(frac=0.01)
    
    X = training_data['text']
    y = training_data['sentiment']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    model_knn = make_pipeline(
        TfidfVectorizer(),
        VotingClassifier([
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                       min_samples_leaf=0.001)),
            ('knn', KNeighborsClassifier(n_neighbors=20))
            
        ])
    )
    
    model_knn.fit(X_train, y_train)
    
    print(model_knn.score(X_train, y_train))
    print(model_knn.score(X_valid, y_valid))
    
    
    #reddit_submissions_summer.select('selftext').show()
    reddit_submissions_summer_text = reddit_submissions_summer.select('selftext').toPandas()
    
    #submissions_summer_predictions = model_knn.predict(reddit_submissions_summer_text['selftext'])
    submissions_summer_predictions = reddit_submissions_summer_text.apply(get_sentiment_vader)
    submissions_summer_predictions = pd.DataFrame(submissions_summer_predictions, columns=['sentiment'])
    submissions_summer_predictions = spark.createDataFrame(submissions_summer_predictions)
    submissions_summer_predictions = submissions_summer_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions_summer = reddit_submissions_summer.join(submissions_summer_predictions, on='id').drop('id')
    
    reddit_submissions_summer.write.json(output + '-submissions-summer', mode='overwrite', compression='gzip')
    
    
    #reddit_comments_summer.select('body').show()
    reddit_comments_summer_text = reddit_comments_summer.select('body').toPandas()
    
    comments_summer_predictions = model_knn.predict(reddit_comments_summer_text['body'])
    comments_summer_predictions = pd.DataFrame(comments_summer_predictions, columns=['sentiment'])
    comments_summer_predictions = spark.createDataFrame(comments_summer_predictions)
    comments_summer_predictions = comments_summer_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_summer = reddit_comments_summer.join(comments_summer_predictions, on='id').drop('id')
    
    reddit_comments_summer.write.json(output + '-comments-summer', mode='overwrite', compression='gzip')
    
    
    reddit_submissions_winter_text = reddit_submissions_winter.select('selftext').toPandas()
    
    submissions_winter_predictions = model_knn.predict(reddit_submissions_winter_text['selftext'])
    submissions_winter_predictions = pd.DataFrame(submissions_winter_predictions, columns=['sentiment'])
    submissions_winter_predictions = spark.createDataFrame(submissions_winter_predictions)
    submissions_winter_predictions = submissions_winter_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions_winter = reddit_submissions_winter.join(submissions_winter_predictions, on='id').drop('id')
    
    reddit_submissions_winter.write.json(output + '-submissions-winter', mode='overwrite', compression='gzip')
    
    
    reddit_comments_winter_text = reddit_comments_winter.select('body').toPandas()
    
    comments_winter_predictions = model_knn.predict(reddit_comments_winter_text['body'])
    comments_winter_predictions = pd.DataFrame(comments_winter_predictions, columns=['sentiment'])
    comments_winter_predictions = spark.createDataFrame(comments_winter_predictions)
    comments_winter_predictions = comments_winter_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_winter = reddit_comments_winter.join(comments_winter_predictions, on='id').drop('id')
    
    reddit_comments_winter.write.json(output + '-comments-winter', mode='overwrite', compression='gzip')
    

input_submissions = sys.argv[1]
input_comments = sys.argv[2]
output = sys.argv[3]
spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '3.5' # make sure we have Spark 3.5+
spark.sparkContext.setLogLevel('WARN')
#sc = spark.sparkContext

main(input_submissions, input_comments, output)