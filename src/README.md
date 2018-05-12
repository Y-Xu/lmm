# The source code of our LMMs
This code is developed based on the classic CBOW model, which is avaiable at https://github.com/dav/word2vec

## Preprocessing

Firstly, preprocess your traning corpus and generate the vocabulary of your corpus.

Secondly, perform an unsupervised morpheme segmentation using Morefessor (http://morpho.aalto.fi/projects/morpho) for the vocabularies.

Then, execute matching between the segmentation results and the morphological compositions in the lookup tables, which can be found in the "../dataset" directory.

Finally replace the matched morphemes with their latent meanings.

## Training

use "make" to compile lmm-a.c lmm-s.c and lmm-m.c

run the script "train_word_embedding.sh" to train word embeddings.
