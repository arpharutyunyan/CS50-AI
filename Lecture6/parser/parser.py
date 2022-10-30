"""
Result 
    https://www.youtube.com/watch?v=ApjkaW6MNJI
    
"""

import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to" | "until"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | NP VP CP
VP -> V | Adv VP | VP NP | VP Adv
NP -> N | Adj NP | Det NP | N P NP | P NP | N Adv | N P S
CP -> Conj S | Conj VP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():
    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    # list with words from sentence
    token = nltk.word_tokenize(sentence)

    words_for_remove = []
    for index in range(len(token)):
        token[index] = token[index].lower()
        # check
        for letter in token[index]:
            # check if in word there is at least one alphabetic character
            if letter.isalpha():
                break
        else:
            words_for_remove.append(token[index])

    # if we have removed word
    if len(words_for_remove) != 0:
        for word in words_for_remove:
            token.remove(word)
    return token


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    list_of_noun_phrase = []
    for i in tree.subtrees():
        if i.label() == 'N':
            list_of_noun_phrase.append(i)
    return list_of_noun_phrase


if __name__ == "__main__":
    main()