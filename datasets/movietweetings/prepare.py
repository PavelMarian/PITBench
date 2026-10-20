"""MovieTweetings (IMDb ratings posted on Twitter), 2013 .. present.

Source: https://github.com/sidooms/MovieTweetings, latest/ (snapshot of the
repository's master branch at download time; sha256 in dataset.json).

Adaptation:
* '::'-separated .dat files -> CSV; unix timestamps -> ISO UTC;
* the ``movies`` title carries the release year in parentheses -> split
  into ``title_year``; genres '|'-joined -> ``movie_genres`` (movie_id, genre);
* ``users`` maps user_id -> twitter_id only; there is no account creation
  time and no movie "release" time in the data: existence of both entities
  is evidenced only by their first rating (entity-existence class).
"""
import pandas as pd

from lib.common import RAW

SRC = RAW / "movietweetings"


def _dat(name, cols, dtype=str):
    return pd.read_csv(SRC / f"{name}.dat", sep="::", engine="python", header=None, names=cols,
                       dtype=dtype, encoding="utf-8", quoting=3)


def prepare():
    ratings = _dat("ratings", ["user_id", "movie_id", "rating", "rating_timestamp"])
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(int)
    ratings["ts"] = pd.to_datetime(ratings["rating_timestamp"].astype("int64"), unit="s")
    ratings = ratings.drop(columns=["rating_timestamp"])
    movies = _dat("movies", ["movie_id", "title", "genres"])
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["title_year"] = pd.to_numeric(movies["title"].str.extract(r"\((\d{4})\)\s*$")[0],
                                         errors="coerce")
    genres = movies[["movie_id", "genres"]].dropna()
    genres = genres.assign(genre=genres["genres"].str.split("|")).explode("genre")[["movie_id", "genre"]]
    movies = movies[["movie_id", "title_year"]]
    users = _dat("users", ["user_id", "twitter_id"])
    users["user_id"] = users["user_id"].astype(int)
    users = users.rename(columns={"twitter_id": "twitter_user_id"})
    return {"ratings": ratings, "movies": movies, "movie_genres": genres, "users": users}
