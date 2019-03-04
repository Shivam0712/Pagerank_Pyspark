
## Source Codes

This repository contains all the codes used in this project.

**1. pagerank.py : Calculates the default pagerank of web pages**

Run Command: pyspark pagerank.py <input_file> <no. of iterations>

**2. pagerank_mod.py: Calculates the pagerank of web pages as a probability distribution**

Run Command: pyspark pagerank_mod.py <input_file> <no. of iterations>

**3. pagerank_articles.py: Calculates and returns pagerank of given articles at different iterations of interest**

Run Command: pyspark pagerank_articles.py <input_file> <no. of iterations> <iterations_of_interest> <articles> <output_path>

**3. pagerank_damping.py: Calculates and returns pagerank for different damping factors**

Run Command: pyspark pagerank_damping.py <input_file> <no. of iterations> <damping_factors> <output_path>
