#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
This is an example implementation of PageRank. For more conventional use,
Please refer to PageRank implementation provided by graphx

Example Usage:
bin/spark-submit examples/src/main/python/pagerank.py data/mllib/pagerank_data.txt 10
"""
from __future__ import print_function

import re
import sys
from operator import add

from pyspark.sql import SparkSession


def computeContribs(urls, rank):
    """Calculates URL contributions to the rank of other URLs."""
    num_urls = len(urls)
    for url in urls:
        yield (url, rank / num_urls)


def parseNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: pagerank <file> <iterations> <interest_iterations> <interest_articles> <output_path> ", file=sys.stderr)
        sys.exit(-1)

    print("WARN: This is a naive implementation of PageRank and is given as an example!\n" +
          "Please refer to PageRank implementation provided by graphx",
          file=sys.stderr)
    
    # import file with iterations of interest 
    iterations_file = open(sys.argv[3])
    
    # initiate list for iterations
    iterations = []
    for line in iterations_file:
        iterations.append(int(line))
        
    # import list with articles of interest
    articles_file = open(sys.argv[4])
    
    # initiate list for articles
    articles = []
    for line in articles_file:
        articles.append(line.split('\r')[0])
    print(articles)

    # initiate output list
    output_list = ['iteration,article,pagerank']
    
    # Initialize the spark context.
    spark = SparkSession\
        .builder\
        .appName("PythonPageRank")\
        .getOrCreate()

    # Loads in input file. It should be in format of:
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     ...
    lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])

    # Loads all URLs from input file and initialize their neighbors.
    links = lines.map(lambda urls: parseNeighbors(urls)).distinct().groupByKey().cache()

    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    ranks = links.map(lambda url_neighbors: (url_neighbors[0], 1.0))

    # Calculates and updates URL ranks continuously using PageRank algorithm.
    for iteration in range(int(sys.argv[2])):
        # Calculates URL contributions to the rank of other URLs.
        contribs = links.join(ranks).flatMap(
            lambda url_urls_rank: computeContribs(url_urls_rank[1][0], url_urls_rank[1][1]))

        # Re-calculates URL ranks based on neighbor contributions.
        ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * 0.85 + 0.15)
        
        if iteration in iterations:
            # Calculate the sum of all page ranks 
            rank_sum = ranks.values().sum()
            
            # Normalize each rank by dividing by total sum of ranks
            new_ranks = ranks.reduceByKey(add).mapValues(lambda rank: rank/rank_sum)
            
            # Subset new_ranks with article of interest
            new_ranks = new_ranks.filter(lambda x: x[0] in articles)
            
            # Collects all URL ranks and dump them to output_list.
            for (link, rank) in new_ranks.collect():
                output_list.append(str(iteration)+','+str(link)+','+str(rank))
    
    print(output_list)
    output_file = open(sys.argv[5],"w")
    output_file.write("\n".join(output_list))
    output_file.close()
    
    spark.stop()
