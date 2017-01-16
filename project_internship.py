from pyspark.mllib.recommendation import ALS
from numpy import array
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from py4j.protocol import Py4JJavaError
import sys
import math

movie_dataset=sys.argv[1]
ratings_dataset=sys.argv[2]

conf = SparkConf().setAppName("Recommedation Systems").set("spark.executor.memory", "2g").setMaster("local[*]")
sc = SparkContext(conf=conf)


#output file

target = open('output.txt', 'w')

# creating a dictionary of the movies to access later for the name of the recommended movies based on ratings
def MovieDict(filepath):
    movie_dic = {}
    count=0
    with open(filepath) as f:
        for line in f:
            #splitting with ',' as it is a CSV
            header = line.split(',')
            #excluding header
            if(header[0]!='movieId'):
                movie_id = int(header[0])
                movie_name = str(header[1])
                movie_dic[movie_id] = movie_name
    return movie_dic

#reading the ratings data
ratingsRdd = sc.textFile(ratings_dataset)
#extracting the header
ratingsRdd_header = ratingsRdd.take(1)[0]


#eliminating header from the data so that while training it is not present and caching it to save memory
ratings_data = ratingsRdd.filter(lambda line: line!=ratingsRdd_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()


#creating a dictionary for the movie dataset for later use of extracting the name of top 'n' movies
movie_dict = MovieDict(movie_dataset)
sc.broadcast(movie_dict)
moviesRdd = sc.textFile(movie_dataset)
moviesRdd_header = moviesRdd.take(1)[0]


movies_data = moviesRdd.filter(lambda line: line!=moviesRdd_header)\
    .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1])).cache()


movie_titles = movies_data.map(lambda x: (int(x[0]), x[1]))

#creating a random split of 60% training, 20% validation, 20% testing datasets
train_RDD, validation_RDD, test_RDD = ratings_data.randomSplit([6,2,2], seed=0L)
#extractig fields required to validate
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))


#setting parameters to train datasets using ALS
seed = 5L
iterations = 10
regularization_parameter = 0.1
#testing for a few ranks to determine which produces the least error
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1

for rank in ranks:
    model = ALS.train(train_RDD, rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err +=1
        print('For rank %s the RMSE is %s' % (rank, error))
        if error < min_error:
            min_error = error
                best_rank = rank

print('The best model was trained with rank %s' % best_rank)

#using the best rank train the training set using ALS
model = ALS.train(train_RDD, best_rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
#call prediction on the testing set using the trained model
predictions = model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
output=rates_and_preds.collect()
t('For testing data the RMSE is %s' % error)

#specifying the user ID to whom you want to recommend the movies
user_id=int(sys.argv[3])
#determining the n
n=int(sys.argv[4])
#returns a list of Rating objects sorted by the predicted rating in descending order
recommendations=model.recommendProducts(user_id, n)
target.write('\nThe recommended movies are:')
for item in recommendations:
    target.write('\n Movie %s with rating \t %.2f' %(movie_dict[item[1]],item[2]))
