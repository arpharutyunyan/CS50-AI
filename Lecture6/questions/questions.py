"""
Result 
    https://www.youtube.com/watch?v=HTAoegF7DzQ
    
"""

import nltk
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    data = {}
    dir_list = os.listdir(directory)
    for filename in dir_list:
        path = os.path.join(directory, f"{filename}")
        with open(path) as f:
            text = f.read()
            data[filename] = text
    return data


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # list with words from document
    token = nltk.word_tokenize(document)
    words_for_remove = []
    for index in range(len(token)):
        token[index] = token[index].lower()
        if token[index] in nltk.corpus.stopwords.words('english'):
            words_for_remove.append(token[index])
        if token[index] in string.punctuation:
            words_for_remove.append(token[index])

    if len(words_for_remove) != 0:
        for word in words_for_remove:
            token.remove(word)

    return token


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    res = {}
    for key in documents:
        text = documents[key]
        for word in text:
            if word in res:
                continue
            else:
                count = 0
                for name in documents:
                    if word in documents[name]:
                        count += 1
                res[word] = math.log(len(documents) / count)
    return res


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf = {}
    for file in files:
        sum = 0
        for word in query:
            idf = idfs[word]
            tf = files[file].count(word)
            sum += tf * idf
        tf_idf[file] = sum
    rank = sorted(tf_idf.keys(), key=lambda x: tf_idf[x], reverse=True)
    rank = list(rank)
    try:
        return rank[:n]
    except:
        return rank


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf = {}
    for sentence in sentences:
        sum = 0
        words = sentences[sentence]
        count = len(words)
        word_count = 0
        for word in query:
            word_count = words.count(word)
            if word in words:
                sum += idfs[word]
        idf[sentence] = (sum, word_count / count)

    rank = sorted(idf.keys(), key=lambda x: idf[x], reverse=True)
    rank = list(rank)
    try:
        return rank[:n]
    except:
        return rank


if __name__ == "__main__":
    main()