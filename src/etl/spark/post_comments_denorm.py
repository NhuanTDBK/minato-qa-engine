from argparse import ArgumentParser
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def main(
    comments_path: str,
    posts_path: str,
    output_path: str,
):
    spark: SparkSession = SparkSession.builder.appName(
        "PostCommentsDenorm"
    ).getOrCreate()
    df_comments = spark.read.format("parquet").load(comments_path)
    df_posts = spark.read.format("parquet").load(posts_path)

    # Comment schema
    # author_flair_text: string
    # author_flair_css_class: string
    # author: string
    # subreddit_id: string
    # body: string
    # id: string
    # name: string
    # created_utc: int64
    # downs: int64
    # subreddit: string
    # ups: int64
    # parent_id: string
    # score: int64

    # Post schema
    # created: int64
    # subreddit: string
    # id: string
    # num_comments: int64
    # score: int64
    # selftext: string
    # title: string
    # thumbnail: string
    # link_flair_type: string
    # link_flair_css_class: string
    # author_flair_css_class: string
    # link_flair_text: string
    # upvote_ratio: double
    # ups: int64
    # over_18: bool
    # is_video: bool

    # add prefix t3_ to id
    df_posts = df_posts.withColumn(
        "post_id", F.concat(F.lit("t3_"), F.col("id"))
    ).filter(
        # (F.col("subreddit") == "Naruto")
        # & (F.col("link_flair_css_class") == "discussion")
        # & (F.col("link_flair_text") == "Question")
        ~F.col("selftext").isin("[deleted]", "[removed]")
    )

    df_agg_comments = (
        df_comments.withColumn(
            "rank",
            F.rank().over(Window.partitionBy("parent_id").orderBy(F.desc("score"))),
        )
        .groupBy("parent_id")
        .agg(
            F.collect_list(F.struct("body", "ups", "score", "downs", "rank")).alias(
                "comments"
            )
        )
    )

    df_joint = df_posts.join(
        df_agg_comments, df_agg_comments.parent_id == df_posts.post_id, how="left"
    ).repartition(10)
    df_joint.write.partitionBy("subreddit").mode("overwrite").parquet(output_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--comments_path", type=str, required=True)
    parser.add_argument("--posts_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    args = parser.parse_args()
    main(
        comments_path=args.comments_path,
        posts_path=args.posts_path,
        output_path=args.output_path,
    )
