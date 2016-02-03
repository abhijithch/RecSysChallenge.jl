
Recommendation System Challenge
=============================

Introduction
---------------

Recommender systems have been playing a pivotal role in various business settings like e-commerce sites, social media platforms, and other platforms involving user interaction with other users or products. Recommender systems provide valuable insights to gain actionable 
intelligence on these users. 

Large Scale Recommender systems help in unraveling the latent information in the complex relational data between users and items. Mapping of the users space to the items space, to predict the interaction is a challenge. Inferring actionable information from variety of data sources collected either implicitly like, click patterns, browser history etc. or explicitly like ratings of books and movies, is what well designed recommender systems do consistently well.

Matrix Factorizations
-------------------------

Depending on the source of information on the users and the items, there are variety of techniques to build recommender systems, each of them having a unique mathematical approach. Linear algebra and matrix factorisations are important to certain types of recommenders where user ratings are available and it is most ideal to apply methods like `svd` in such cases.

In matrix factorization the users and items are mapped onto a joint latent factor space of reduced dimension f, and the inner product of the user vector with item vector gives the corresponding interaction. Matrix factorization is mainly about a more compact representation of the large training data which is obtained by dimensionality reduction. We want to quantify the nature or the characteristics of the movies defined by a certain number of aspects (factors), i.e we are trying to generalize the information (independent and unrelated ratings matrix) in a concise and descriptive way. 


> **Example :**
Let us consider a simple example to figure out how matrix factorization helps in predicting the likelihood of a user liking a movie or not. 
For sake of brevity, we have couple of users, Joe and Jane and couple of movies, Titanic and Troll 2. The users and the movies are characterized based on certain number of factors as show in the below tables. 

| Factors/Movies   |      Titanic      |  Troll 2 |
|:----------|-------------:|------:|
| Romance |  4 | 1 |
| Comedy |    2  |   4 |
| Box Office success | 5 | 2 |
| Drama |    3  |  2 |
| Horror| 1 |    4 |


| Factors/Movies   |      Joe      |  Jane |
|:----------|-------------:|------:|
| Romance |  4 | 1 |
| Comedy |    2  |   4 |
| Box Office success | 5 | 2 |
| Drama |    3  |  2 |
| Horror| 1 |    4 |


Consider Joe to be characterized by vector `[4 2 5 3 1]`,  which suggests that Joe likes Romance and big hit movies and not so much horror or comedy. Similarly Jane likes comedy horror and she is not very particular about box office success of the movies, neither is she a big fan of romance movies. 

The movies Titanic, is a popular romance movie, where as the movie Troll 2, is not so popular and horror comedy. It is intuitively obvious that Joe will end up liking Titanic and Jane will like Troll 2. This is based on how the users and movies score on the 5 factors. Using *Cosine distance* as shown in the below table, confirms this. 
$$
\cos(\theta) = \frac{Titanic * Joe}{\|Titanic\|\|Joe\|}
$$

| Factors/Movies   |      Joe      |  Jane |
|:----------|-------------:|------:|
| Titanic |  0.94 | 0.67 |
| Troll 2 |  0.50  | 0.97 |

With large rating data matrix, like in the NETFLIX dataset which had around 20 thousand movies and 0.5 million users, mapping all the users and the movies in the above way is impossible. This is where matrix factorization helps in factoring the Rating matrix into user matrix and movie matrix. 

Alternating Least Squares
-------------------------------

![enter image description here](https://lh3.googleusercontent.com/-cwmHoKDAXN8/Vouil5EDRLI/AAAAAAAABMc/r2AQobqti-o/s0/ALS_FIG.jpg "ALS_FIG.jpg")

Let $U={u_i}$ be the user feature matrix where ${u_i} \subseteq
\mathbb{R}^{n_f}$ and $i=1,2,...,n_u$, and let $M={m_j}$ be the item or
movie feature matrix, where ${m_j} \subseteq \mathbb{R}^{n_f}$ and $j=1,2,...,n_m$. Here $n_f$ is the number of factors, i.e., the reduced dimension or the lower rank, which is determined by cross validation. The predictions can be calculated for any user-movie combination, 
$(i,j)$, as $r_{ij}={u_i} \cdotp {m_j}, \forall i,j$. 

Here we minimize the loss function of $U$ and $M$ as the condition in the iterative process of obtaining these matrices. Let us start by considering the loss due to a single prediction in terms of squared error: 
\begin{equation}
 \mathcal{L}^2(r,{u},{m})=(r-<{u},{m}>)^2.
\end{equation}

Based on the above equation generalizing it for the whole data set, the
\emph{empirical} total loss as:
\begin{equation}
 \mathcal{L}^{emp}(R,U,M)=\frac{1}{n} \sum_{(i,j) \in
I}\mathcal{L}^2(r_{ij},{u_i},{m_j}),
\end{equation}
where $I$ is the known ratings dataset having $n$ ratings. 

Julia recommender system 
-------------------------------
The package [RecSys.jl](https://github.com/abhijithch/RecSys.jl/) is a package for recommender systems in Julia, it can currently work with explicit ratings data. The API for preparing the input is creating an instance of `ALSWR` type, which expects as input parameters input file location. The second optional input is the variable `par` which specifies the type of parallelism. The default is set to shared memory parallelism, however by passing `par=ParThreads()`, we could set to multithreaded parallelism. 

`rec=ALSWR("/location/to/input/file/File.delim", par=ParThread)`

The file can be any tabular structured data, delimited by any character, which needs to be specified, 

`inp=DlmFile(name::AbstractString; dlm::Char=Base.DataFmt.invalid_dlm(Char), header::Bool=false, quotes::Bool=true)`

The call to the function to create a model is `train(rec, 10, 4)` where 10 is the number of iterations to run and 4 is the number of factors. 

A log of a test run on the NETLIX data is shown below, 

`05-Jan 17:07:28:DEBUG:root:loading inputs...
05-Jan 17:07:50:DEBUG:root:time to load inputs: 22.235926866531372 secs
05-Jan 17:07:50:DEBUG:root:preparing inputs...
05-Jan 17:08:00:DEBUG:root:prep time: 9.460721015930176
05-Jan 17:08:25:DEBUG:root:begin iteration 1
05-Jan 17:08:38:DEBUG:root:	users
05-Jan 17:08:44:DEBUG:root:	items
05-Jan 17:08:44:DEBUG:root:begin iteration 2
05-Jan 17:08:56:DEBUG:root:	users
05-Jan 17:09:01:DEBUG:root:	items
05-Jan 17:09:01:DEBUG:root:begin iteration 3
05-Jan 17:09:13:DEBUG:root:	users
05-Jan 17:09:19:DEBUG:root:	items
05-Jan 17:09:19:DEBUG:root:begin iteration 4
05-Jan 17:09:31:DEBUG:root:	users
05-Jan 17:09:37:DEBUG:root:	items
05-Jan 17:09:37:DEBUG:root:begin iteration 5
05-Jan 17:09:49:DEBUG:root:	users
05-Jan 17:09:54:DEBUG:root:	items
05-Jan 17:09:54:DEBUG:root:begin iteration 6
05-Jan 17:10:07:DEBUG:root:	users
05-Jan 17:10:12:DEBUG:root:	items
05-Jan 17:10:12:DEBUG:root:begin iteration 7
05-Jan 17:10:24:DEBUG:root:	users
05-Jan 17:10:30:DEBUG:root:	items
05-Jan 17:10:30:DEBUG:root:begin iteration 8
05-Jan 17:10:42:DEBUG:root:	users
05-Jan 17:10:48:DEBUG:root:	items
05-Jan 17:10:48:DEBUG:root:begin iteration 9
05-Jan 17:11:00:DEBUG:root:	users
05-Jan 17:11:05:DEBUG:root:	items
05-Jan 17:11:05:DEBUG:root:begin iteration 10
05-Jan 17:11:18:DEBUG:root:	users
05-Jan 17:11:23:DEBUG:root:	items
05-Jan 17:11:23:DEBUG:root:fact time 203.25707387924194
05-Jan 17:12:43:DEBUG:root:rmse time 79.78545188903809
rmse of the model: 0.8593418667702695`

As we can see from the above log, that the `rmse` is 0.8593. This is very good accuracy compared to some of the prize winning models. The timings too are impressive, for data as large as 1 billion ratings. 

Apart from methods to model the data and check for accuracy, there are also abilities to make recommendations for users who have not interacted with items, by picking the most likely items the user would interact with. Hence in RecSys.jl we have a fast, scalable and accurate recommender system which can be used to for end to end system. Upcoming would be a demo of such a system with the UI design too done in Julia, which is possible as of today. 

Recommender System challenge in Julia
------------------------------------------------

To participate in the challenge, please clone the package, [git@github.com:abhijithch/RecSysChallenge.jl.git](git@github.com:abhijithch/RecSysChallenge.jl.git). The data for the first part of the challenge can be downloaded from [http://grouplens.org/datasets/movielens/](http://grouplens.org/datasets/movielens/). Download the 20 million dataset, extract the files into data folder in the repository. We are using the 20 million ratings data, which consists of 20 million ratings for 27,000 movies by 138,000 users. This 20 million ratings can be used as the training set, the students are free to create their test set from the above. However from the competitions point of view, to compute the final RMSE, we would be using a secret set which would not be provided. 

Challenge #1 
----------------

The first challenge would be to come up with an algorithm which would be to outperform the presented methods from the RecSys package. It is expected to beat the RMSE which is computed on a subset of the data which will not be released to the competitors. 

The current implementation is based on the article, *Large-scale Parallel Collaborative Filtering for the Netflix Prize*. 

### Target performance :
The minimum performance numbers expected to qualify for the next challenge is 

| Method/Parameters  |      Prediction Time      |  RMSE |
|:----------|-------------:|------:|
| Sequential |  200 sec | 0.8420 |



Challenge #2 
----------------
One of the biggest challenges in Recommender systems is **scaling**, coping with ever-growing data and tuning the contextual model parameters. 

The next challenge is to apply the above model on a very large dataset. Yahoo has recently released a dataset which is around 13.5 TB(uncompressed). The dataset stands at a massive ~110B events (13.5TB uncompressed) of anonymised user-news item interaction data, collected by recording the user-news item interactions of about 20M users from February 2015 to May 2015. 

Challenge #3 
---------------

So far the main concept we have dealt with is model based Collaborative filtering which unravels latent information with help of matrix factorization techniques like ALS, SVD. However there exists multitude of other techniques and approaches where one can utilise auxiliary data on the items and users, other than the ratings information. With the recent progress in the field of deep learning, it is worthwhile to utilise the all the available information on the users and items and create an ensemble model. In the yahoo data set apart from the reader and news article interaction data, we also have loads of meta data on the users. 

The third challenge is to come up with any novel technique not only based on matrix factorization to further improve on the RMSE by atleast 10%.   


