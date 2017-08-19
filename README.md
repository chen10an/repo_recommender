# repo_recommender
A small recommendation system for GitHub repositories based on collaborative filtering.

Notebook for creating the dataset via GitHub's API and python's requests library:
[make dataset](http://nbviewer.jupyter.org/github/chen10an/repo_recommender/blob/master/make%20dataset.ipynb)

Note: I have not included auth.py in this repo because it contains my login information.
The file can be created with this content:
```py
user = 'username'
pw = 'password'
```

Notebook that contains the recommendation system:
[Collaborative Filtering for GitHub Stars](http://nbviewer.jupyter.org/github/chen10an/repo_recommender/blob/master/Collaborative%20Filtering%20for%20GitHub%20Stars.ipynb)

I have a more detailed notebook for the collaborative filtering algorithm
in my ML algorithms [repository][other repo]. user.py and utils.py are based on this
notebook and are also present in the algorithms repository.

[other repo]: https://github.com/chen10an/ml_algos_python/tree/master/collaborative_filtering
