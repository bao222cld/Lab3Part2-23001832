
import sys, os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from representations.word_embedder import WordEmbedder
import numpy as np

def main():
    e = WordEmbedder("glove-wiki-gigaword-50")
    v = e.get_vector("king")
    print("Vector for 'king' length:", None if v is None else len(v))
    print("Sim king-queen:", e.get_similarity("king", "queen"))
    print("Sim king-man:", e.get_similarity("king", "man"))
    print("Top 10 similar to 'computer':", e.get_most_similar("computer", 10))
    print("Embed 'The queen rules the country.':", e.embed_document("The queen rules the country.").shape)

if __name__ == '__main__':
    main()
