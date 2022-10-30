"""
Result 
    https://www.youtube.com/watch?v=oZlzOBIS-sU
    
"""

import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    trans_model = {}
    for key in corpus:
        trans_model[key] = (1 - damping_factor) / len(corpus)

    if len(corpus[page]) == 0:
        for key in trans_model:
            trans_model[key] = damping_factor / len(corpus)
        return trans_model

    for key in corpus[page]:
        trans_model[key] += damping_factor / len(corpus[page])
    return trans_model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    all_pages = list(corpus.keys())
    # to get first random page
    random_page = random.choice(all_pages)
    sample = transition_model(corpus, random_page, damping_factor)
    res = {}
    # to get all pages in corpus
    for key in corpus:
        res[key] = 0
    for i in range(n):
        for key in sample:
            res[key] += sample[key]
        # getting relative weights
        weights = list(sample.values())
        random_page = random.choices(all_pages, weights, k=1)[0]
        sample = transition_model(corpus, random_page, damping_factor)
    sum_values = sum(res.values())
    for key in res:
        res[key] /= sum_values
    return res


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    dif_ranks = np.array([])
    curr_ranks = np.array([])

    for item in range(len(corpus)):
        dif_ranks = np.append(dif_ranks, 1 / len(corpus))
        # add ranks in sequence of corpus
        curr_ranks = np.append(curr_ranks, 1 / len(corpus))

    res = {}
    first_value = (1 - damping_factor) / len(corpus)

    while np.any(abs(dif_ranks) > 0.001):
        new_ranks = np.array([])
        for page in corpus:
            second_value = 0
            for key, link in corpus.items():
                if page in link:
                    num_links = len(link)
                    # get rank with index, because in list adding ranks in that sequence
                    second_value += (curr_ranks[list(corpus.keys()).index(key)] / num_links)
                elif len(link) == 0:
                    second_value += (curr_ranks[list(corpus.keys()).index(key)] / len(corpus))
            new_ranks = np.append(new_ranks, first_value + (damping_factor * second_value))
        dif_ranks = curr_ranks - new_ranks
        curr_ranks = np.copy(new_ranks)

    index = 0
    for i in corpus:
        res[i] = curr_ranks[index]
        index += 1
    return res


if __name__ == "__main__":
    main()