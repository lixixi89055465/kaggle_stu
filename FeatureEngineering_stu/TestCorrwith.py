import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
# from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors._nearest_centroid import NearestCentroid
import pandas as pd
import pandas as pd
import numpy as np

data = np.array([[5, 5, 3, 3, 4], [3, 4, 5, 5, 4],
                 [3, 4, 3, 4, 5], [5, 5, 3, 4, 4]])
df = pd.DataFrame(data, columns=['The Shawshank Redemption',
                                 'Forrest Gump', 'Avengers: Endgame',
                                 'Iron Man', 'Titanic '],
                  index=['user1', 'user2', 'user3', 'user4'])
# Compute correlation between user1 and other users
user_to_compare = df.iloc[0]
similarity_with_other_users = df.corrwith(user_to_compare, axis=1,
                                          method='pearson')
similarity_with_other_users = similarity_with_other_users.sort_values(
    ascending=False)
# Compute correlation between 'The Shawshank Redemption' and other movies
movie_to_compare = df['The Shawshank Redemption']
similarity_with_other_movies = df.corrwith(movie_to_compare, axis=0)
similarity_with_other_movies = similarity_with_other_movies.sort_values(
    ascending=False)