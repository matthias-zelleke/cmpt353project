import sys
assert sys.version_info >= (3, 10) # make sure we have Python 3.10+
from pyspark.sql import SparkSession, functions, types
import pandas as pd
from datasets import load_dataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


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


def main(input_submissions, input_comments, output):
    
    reddit_submissions_data = spark.read.json(input_submissions, schema=submissions_schema)
    reddit_comments_data = spark.read.json(input_comments, schema=comments_schema)
    
    
    ### Analysis #2, Comparing results between all 4 seasons:
    
    summer_months = [6, 7, 8]
    fall_months = [9, 10, 11]
    winter_months = [12, 1, 2]
    spring_months = [3, 4, 5]
    
    reddit_submissions_summer = reddit_submissions_data.where(functions.col('month').isin(summer_months))
    reddit_submissions_summer = reddit_submissions_summer.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_summer = reddit_comments_data.where(functions.col('month').isin(summer_months))
    reddit_comments_summer = reddit_comments_summer.withColumn('id', functions.monotonically_increasing_id())
    
    
    reddit_submissions_fall = reddit_submissions_data.where(functions.col('month').isin(fall_months))
    reddit_submissions_fall = reddit_submissions_fall.withColumn('id', functions.monotonically_increasing_id())

    reddit_comments_fall = reddit_comments_data.where(functions.col('month').isin(fall_months))
    reddit_comments_fall = reddit_comments_fall.withColumn('id', functions.monotonically_increasing_id())


    reddit_submissions_winter = reddit_submissions_data.where(functions.col('month').isin(winter_months))
    reddit_submissions_winter = reddit_submissions_winter.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_winter = reddit_comments_data.where(functions.col('month').isin(winter_months))
    reddit_comments_winter = reddit_comments_winter.withColumn('id', functions.monotonically_increasing_id())
    
    
    reddit_submissions_spring = reddit_submissions_data.where(functions.col('month').isin(spring_months))
    reddit_submissions_spring = reddit_submissions_spring.withColumn('id', functions.monotonically_increasing_id())

    reddit_comments_spring = reddit_comments_data.where(functions.col('month').isin(spring_months))
    reddit_comments_spring = reddit_comments_spring.withColumn('id', functions.monotonically_increasing_id())

    
    training_data = load_dataset('sentiment140', trust_remote_code=True)['train']
    training_data = training_data.to_pandas()
    training_data = training_data.sample(frac=0.01)
    
    X = training_data['text']
    y = training_data['sentiment']
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    
    model_svc = make_pipeline(
        TfidfVectorizer(),
        SVC(C=0.1, kernel='linear')
    )
    
    model_svc.fit(X_train, y_train)
    
    print(model_svc.score(X_train, y_train))
    print(model_svc.score(X_valid, y_valid))
    
    
    #reddit_submissions_summer.select('selftext').show()
    reddit_submissions_summer_text = reddit_submissions_summer.select('selftext').toPandas()
    
    submissions_summer_predictions = model_svc.predict(reddit_submissions_summer_text['selftext'])
    submissions_summer_predictions = pd.DataFrame(submissions_summer_predictions, columns=['sentiment'])
    submissions_summer_predictions = spark.createDataFrame(submissions_summer_predictions)
    submissions_summer_predictions = submissions_summer_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions_summer = reddit_submissions_summer.join(submissions_summer_predictions, on='id').drop('id')
    
    reddit_submissions_summer.write.json(output + '-submissions-summer', mode='overwrite', compression='gzip')
    
    #reddit_comments_summer.select('body').show()
    reddit_comments_summer_text = reddit_comments_summer.select('body').toPandas()
    
    comments_summer_predictions = model_svc.predict(reddit_comments_summer_text['body'])
    comments_summer_predictions = pd.DataFrame(comments_summer_predictions, columns=['sentiment'])
    comments_summer_predictions = spark.createDataFrame(comments_summer_predictions)
    comments_summer_predictions = comments_summer_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_summer = reddit_comments_summer.join(comments_summer_predictions, on='id').drop('id')
    
    reddit_comments_summer.write.json(output + '-comments-summer', mode='overwrite', compression='gzip')
    
    
    reddit_submissions_fall_text = reddit_submissions_fall.select('selftext').toPandas()

    submissions_fall_predictions = model_svc.predict(reddit_submissions_fall_text['selftext'])
    submissions_fall_predictions = pd.DataFrame(submissions_fall_predictions, columns=['sentiment'])
    submissions_fall_predictions = spark.createDataFrame(submissions_fall_predictions)
    submissions_fall_predictions = submissions_fall_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions_fall = reddit_submissions_fall.join(submissions_fall_predictions, on='id').drop('id')
    
    reddit_submissions_fall.write.json(output + '-submissions-fall', mode='overwrite', compression='gzip')
    
    reddit_comments_fall_text = reddit_comments_fall.select('body').toPandas()
    
    comments_fall_predictions = model_svc.predict(reddit_comments_fall_text['body'])
    comments_fall_predictions = pd.DataFrame(comments_fall_predictions, columns=['sentiment'])
    comments_fall_predictions = spark.createDataFrame(comments_fall_predictions)
    comments_fall_predictions = comments_fall_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_fall = reddit_comments_fall.join(comments_fall_predictions, on='id').drop('id')
    
    reddit_comments_fall.write.json(output + '-comments-fall', mode='overwrite', compression='gzip')

    
    reddit_submissions_winter_text = reddit_submissions_winter.select('selftext').toPandas()
    
    submissions_winter_predictions = model_svc.predict(reddit_submissions_winter_text['selftext'])
    submissions_winter_predictions = pd.DataFrame(submissions_winter_predictions, columns=['sentiment'])
    submissions_winter_predictions = spark.createDataFrame(submissions_winter_predictions)
    submissions_winter_predictions = submissions_winter_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions_winter = reddit_submissions_winter.join(submissions_winter_predictions, on='id').drop('id')
    
    reddit_submissions_winter.write.json(output + '-submissions-winter', mode='overwrite', compression='gzip')
    
    reddit_comments_winter_text = reddit_comments_winter.select('body').toPandas()
    
    comments_winter_predictions = model_svc.predict(reddit_comments_winter_text['body'])
    comments_winter_predictions = pd.DataFrame(comments_winter_predictions, columns=['sentiment'])
    comments_winter_predictions = spark.createDataFrame(comments_winter_predictions)
    comments_winter_predictions = comments_winter_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_winter = reddit_comments_winter.join(comments_winter_predictions, on='id').drop('id')
    
    reddit_comments_winter.write.json(output + '-comments-winter', mode='overwrite', compression='gzip')
    
    
    reddit_submissions_spring_text = reddit_submissions_spring.select('selftext').toPandas()

    submissions_spring_predictions = model_svc.predict(reddit_submissions_spring_text['selftext'])
    submissions_spring_predictions = pd.DataFrame(submissions_spring_predictions, columns=['sentiment'])
    submissions_spring_predictions = spark.createDataFrame(submissions_spring_predictions)
    submissions_spring_predictions = submissions_spring_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_submissions_spring = reddit_submissions_spring.join(submissions_spring_predictions, on='id').drop('id')
    
    reddit_submissions_spring.write.json(output + '-submissions-spring', mode='overwrite', compression='gzip')
    
    
    reddit_comments_spring_text = reddit_comments_spring.select('body').toPandas()
    
    comments_spring_predictions = model_svc.predict(reddit_comments_spring_text['body'])
    comments_spring_predictions = pd.DataFrame(comments_spring_predictions, columns=['sentiment'])
    comments_spring_predictions = spark.createDataFrame(comments_spring_predictions)
    comments_spring_predictions = comments_spring_predictions.withColumn('id', functions.monotonically_increasing_id())
    
    reddit_comments_spring = reddit_comments_spring.join(comments_spring_predictions, on='id').drop('id')
    
    reddit_comments_spring.write.json(output + '-comments-spring', mode='overwrite', compression='gzip')



input_submissions = sys.argv[1]
input_comments = sys.argv[2]
output = sys.argv[3]
spark = SparkSession.builder.appName('example code').getOrCreate()
assert spark.version >= '3.5' # make sure we have Spark 3.5+
spark.sparkContext.setLogLevel('WARN')
#sc = spark.sparkContext

main(input_submissions, input_comments, output)