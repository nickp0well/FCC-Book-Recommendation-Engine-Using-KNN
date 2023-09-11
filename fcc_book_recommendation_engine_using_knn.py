# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# Clean the data, remove from the dataset users with less than 200 ratings and books with less than 100 ratings.
userCounts = df_ratings['user'].value_counts()
isbnCounts = df_ratings['isbn'].value_counts()
#remove all users with less than 200 reviews
df_ratings = df_ratings[~df_ratings['user'].isin(userCounts[userCounts < 200].index)]
#remove all books with less than 100 ratings
df_ratings = df_ratings[~df_ratings['isbn'].isin(isbnCounts[isbnCounts < 100].index)]

# Combine book data with rating data
df_combined = pd.merge(df_books, df_ratings, on = 'isbn')
columns = ['author']
df_combined = df_combined.drop(columns, axis=1)

# Group by book titles, and create a new column for total rating count
book_ratingCount = (df_combined.
                    groupby(by = ['title'])['rating'].
                    count().
                    reset_index().
                    rename(columns = {'rating': 'totalRatingCount'})
                    [['title', 'totalRatingCount']]
                  )

# Combine rating data with total rating count data and convert table to 2D matrix, and fill missing values with zeroes
rating_with_totalRatingCount = df_combined.merge(book_ratingCount, left_on='title', right_on='title', how='inner')
rating_with_totalRatingCount = rating_with_totalRatingCount.drop_duplicates(['title', 'user'])
rating_with_totalRatingCount_pivot = rating_with_totalRatingCount.pivot(index = 'title', columns = 'user', values = 'rating').fillna(0)
rating_with_totalRatingCount_matrix = csr_matrix(rating_with_totalRatingCount_pivot.values)

# Create the KNN model
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(rating_with_totalRatingCount_matrix)

# function to return recommended books - this will be tested
def get_recommends(book = ""):
  x=rating_with_totalRatingCount_pivot.loc[book].array.reshape(1, -1)
  distances,indices=model_knn.kneighbors(x,n_neighbors=6)
  R_books=[]
  for distance,indice in zip(distances[0],indices[0]):
    if distance!=0:
      R_book=rating_with_totalRatingCount_pivot.index[indice]
      R_books.append([R_book,distance])
  recommended_books=[book,R_books[::-1]]
  return recommended_books

books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)

def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2):
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You haven't passed yet. Keep trying!")

test_book_recommendation()
