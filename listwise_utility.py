####
# This is an adaption of TensorFlow Recommenders
# listwise utility source code that is licensed under
# an Apache 2.0 license bearing the following copywrite:
# Copyright 2022 The TensorFlow Recommenders Authors
####

import array
import collections

from typing import Dict, List, Optional, Text, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs


def evaluate(user_model: tf.keras.Model,
             movie_model: tf.keras.Model,
             test: tf.data.Dataset,
             movies: tf.data.Dataset,
             train: Optional[tf.data.Dataset] = None,
             k: int = 10) -> Dict[Text, float]:
    """Evaluates a Movielens model on the supplied datasets.

    Args:
        user_model: User representation model.
        movie_model: Movie representation model.
        test: Test dataset.
        movies: Dataset of movies.
        train: Training dataset. If supplied, recommendations for training watches
          will be removed.
        k: The cutoff value at which to compute precision and recall.

    Returns:
        Dictionary of metrics.
    """

    book_ids = np.concatenate(
      list(books.batch(1000).as_numpy_iterator()))

    book_vocabulary = dict(zip(book_ids.tolist(), range(len(book_ids))))

    train_user_to_books = collections.defaultdict(lambda: array.array("i"))
    test_user_to_books = collections.defaultdict(lambda: array.array("i"))

    if train is not None:
        for row in train.as_numpy_iterator():
            user_id = row[1]
            book_id = book_vocabulary[row[0]]
            train_user_to_books[user_id].append(book_id)

    for row in test.as_numpy_iterator():
        user_id = row[1]
        book_id = book_vocabulary[row[0]]
        train_user_to_books[user_id].append(book_id)

    book_embeddings = np.concatenate(
      list(books.batch(4096).map(
          lambda x: book_model(x[1])
      ).as_numpy_iterator()))

    precision_values = []
    recall_values = []

    for user_id, test_books in test_user_to_books.items():
        user_embedding = user_model(np.array([user_id])).numpy()
        scores = (user_embedding @ book_embeddings.T).flatten()

        test_movies = np.frombuffer(test_books, dtype=np.int32)

        if train is not None:
            train_books = np.frombuffer(
              train_user_to_movies[user_id], dtype=np.int32)
            scores[train_books] = -1e6

        top_books = np.argsort(-scores)[:k]
        num_test_books_in_k = sum(x in top_books for x in test_books)
        precision_values.append(num_test_books_in_k / k)
        recall_values.append(num_test_books_in_k / len(test_books))

    return {
      "precision_at_k": np.mean(precision_values),
      "recall_at_k": np.mean(recall_values)
    }


def _create_feature_dict() -> Dict[Text, List[tf.Tensor]]:
    """Helper function for creating an empty feature dict for defaultdict."""
    return {"book_title": [], "user_rating": []}


def _sample_list(
    feature_lists: Dict[Text, List[tf.Tensor]],
    num_examples_per_list: int,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Function for sampling a list example from given feature lists."""
    if random_state is None:
        random_state = np.random.RandomState()

    sampled_indices = random_state.choice(
      range(len(feature_lists["book_title"])),
      size=num_examples_per_list,
      replace=False,
    )
    sampled_book_titles = [feature_lists["book_title"][idx] for idx in sampled_indices]
    sampled_ratings = [feature_lists["user_rating"][idx] for idx in sampled_indices]

    return (
      tf.concat(sampled_book_titles, 0),
      tf.concat(sampled_ratings, 0),
    )


def sample_listwise(
    rating_dataset: tf.data.Dataset,
    num_list_per_user: int = 10,
    num_examples_per_list: int = 10,
    seed: Optional[int] = None,
) -> tf.data.Dataset:
    """Function for converting the MovieLens 100K dataset to a listwise dataset.

    Args:
      rating_dataset:
        The MovieLens ratings dataset loaded from TFDS with features
        "movie_title", "user_id", and "user_rating".
      num_list_per_user:
        An integer representing the number of lists that should be sampled for
        each user in the training dataset.
      num_examples_per_list:
        An integer representing the number of movies to be sampled for each list
        from the list of movies rated by the user.
      seed:
        An integer for creating `np.random.RandomState`.

    Returns:
      A tf.data.Dataset containing list examples.

      Each example contains three keys: "user_id", "movie_title", and
      "user_rating". "user_id" maps to a string tensor that represents the user
      id for the example. "movie_title" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.string. It represents the list
      of candidate movie ids. "user_rating" maps to a tensor of shape
      [sum(num_example_per_list)] with dtype tf.float32. It represents the
      rating of each movie in the candidate list.
    """
    random_state = np.random.RandomState(seed)

    example_lists_by_user = collections.defaultdict(_create_feature_dict)

    book_title_vocab = set()
    for example in rating_dataset:
        user_id = example[0].numpy()
        example_lists_by_user[user_id]["book_title"].append(
            example[1])
        example_lists_by_user[user_id]["user_rating"].append(
            example[2])
        book_title_vocab.add(example[1].numpy())

    tensor_slices = {"user_id": [], "book_title": [], "user_rating": []}

    for user_id, feature_lists in example_lists_by_user.items():
        for _ in range(num_list_per_user):

        # Drop the user if they don't have enough ratings.
            if len(feature_lists["book_title"]) < num_examples_per_list:
                continue

            sampled_book_titles, sampled_ratings = _sample_list(
              feature_lists,
              num_examples_per_list,
              random_state=random_state,
            )
            tensor_slices["user_id"].append(user_id)
            tensor_slices["book_title"].append(sampled_book_titles)
            tensor_slices["user_rating"].append(sampled_ratings)

    return tf.data.Dataset.from_tensor_slices(tensor_slices)



class RankingModel(tfrs.Model):
    def __init__(self, loss, user_model, book_model):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = user_model
        # Compute embeddings for movies.
        self.book_embeddings = book_model

        # Feed embeddings
        self.score_model = tf.keras.Sequential([
          # Learn multiple dense layers.
          tf.keras.layers.Dense(256, activation="relu"),
          tf.keras.layers.Dense(64, activation="relu"),
          # Make rating predictions in the final layer.
          tf.keras.layers.Dense(1)
        ])

        self.task = tfrs.tasks.Ranking(
          loss=loss,
          metrics=[
            tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
            tf.keras.metrics.RootMeanSquaredError()
          ]
        )

    def call(self, features):
        # We first convert the id features into embeddings.
        # User embeddings are a [batch_size, embedding_dim] tensor.
        user_embeddings = self.user_embeddings(features["user_id"])

        # Movie embeddings are a [batch_size, num_movies_in_list, embedding_dim]
        # tensor.
        book_embeddings = self.book_embeddings(features["book_title"])

        # We want to concatenate user embeddings with movie emebeddings to pass
        # them into the ranking model. To do so, we need to reshape the user
        # embeddings to match the shape of movie embeddings.
        list_length = features["book_title"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1)

        # Once reshaped, we concatenate and pass into the dense layers to generate
        # predictions.
        concatenated_embeddings = tf.concat(
            [user_embedding_repeated, book_embeddings], 2)

        return self.score_model(concatenated_embeddings)

    def compute_loss(self, features, training=False):
        labels = tf.strings.to_number(features.pop("user_rating"))
        scores = self(features)
        
        return self.task(labels=labels, predictions=tf.squeeze(scores, axis=-1),)