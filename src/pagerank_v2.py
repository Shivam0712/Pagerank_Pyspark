
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
    if len(sys.argv) != 4:
        
        print("Usage: pagerank <file> <iterations> <damping>", file=sys.stderr)
        print("<file>: Input File,\ 
              \n<iterations>: File with list of iterations,\ 
              \n<damping>: File with list of damping factor")
        sys.exit(-1)

    print("WARN: This is a naive implementation of PageRank and is given as an example!\n" +
          "Please refer to PageRank implementation provided by graphx",
          file=sys.stderr)
    
    # import file with iterations of interest 
    iterations_file = open(sys.argv[2])
    
    # initiate list for iterations
    iterations = []
    for line in iterations_file:
        iterations.append(int(line))
        
    max_iter = max(iterations)
    
    # import file with iterations of interest 
    damping_file = open(sys.argv[3])
    
    # initiate list for iterations
    damping = []
    for line in damping_file:
        damping.append(float(line))
     
    # initiate output list
    output_list = ['iterations,damping,pagerank,inputlinks']
    
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

    # Run loop over damping
    for d in damping:
        
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
            ranks = contribs.reduceByKey(add).mapValues(lambda rank: rank * d + (1 - d))
            
            # save out put
            if iteration in iterations:
                
                # Calculate the sum of all page ranks
                rank_sum = ranks.values().sum()

                # Normalize each rank by dividing by total sum of ranks
                new_ranks = ranks.reduceByKey(add).mapValues(lambda rank: rank/rank_sum)

                # count input links
                input_links = contribs.countByKey()

                # Collects all URL ranks and dump them to output_list.
                for (link, rank) in new_ranks.collect():
                    output_list.append(str(iteration)+','+str(d)+','+str(rank)+','+str(input_links[link]))

    output_file = open('output.txt',"w")
    output_file.write("\n".join(output_list))
    output_file.close()
    
    spark.stop()
