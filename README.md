# SentenceCompletion

Microsoft Sentence Completion Challenge is a benchmark proposed by researchers at Microsoft. This challenge provides a common measurement technique for testing the performance of different models developed using distinct methodologies. In this research, three approaches have been implemented which are based on traditional n-gram models, word embeddings similarity and deep learning models. For each approach, multiple models have been implemented by varying the hyperparameters and finally, the optimised models for each approach have been compared against each other. The models have also been tested for statistical metrics such as accuracy, entropy, and uncertainty. Deep learning models outperformed all other approaches and BERT specifically had the highest accuracy out of all the models.

# Language modelling

## N - gram language modelling
Traditional N - gram models use the Markov assumption where the probability of a word is dependent on the previous n-1 words. Where n is the number of previous words to be considered, i is the index of the current word, and w is word under consideration. For experimentation with the MSCC data, unigram (n = 1), bigram (n = 2) and trigram (n = 3) models which have been trained on the Sherlock Holmes dataset. The training data has been increased incrementally to find its performance on the testing data. 

## Word embedding model
Word embeddings are high dimensional spaces which represent words as a point in them. This method provides a way of converting words into vectors. Latent Semantic Analysis is a technique which uses these vectors to model semantics. One of the widely used word embeddings is word2vec proposed by (Mikolov et al). This method combines the usage of skip-gram, negative sampling, and transformers to generate efficient word embeddings based on their respective context. In this research, two word embeddings have been used individually as well as in unison which are, word2vec and fasttext. Cosine similarity has been used to measure the distance between the two vectors and their respective similarities. The average of two embeddings is also tested for.

## Deep learning models
Language modelling has recently shifted its focus to neural based models from count-based models. The order in which the tokens appear, and representation of the hidden states are taken into consideration. At the beginning of a new sentence, these conditions are reset. 


### BERT
This model consists of an embedding layer which, a transformer encoder, a fully connected classification layer and finally a softmax layer which further converts the probable embeddings to words. The transformers consist of connections which are residual and multi-head self-attention. BERT is created by tokenization using wordpiece which uses special tokens such as “CLS” (start of the sequence) and “SEP” (end of the sequence). BERT has two modelling objectives namely: masked language model and next sequence prediction (NSP). These modelling objectives serve as the base for our usage in this paper [14]. The model was trained on over 16 GB of training data. The creators of BERT have introduced two types of BERTs based on their sizes. BERTBASE is the first model which has 12 transformer blocks, 768 hidden size and 12 self-attention heads while for BERTLARGE, there are 24 transformer blocks, 1024 hidden size and 16 self-attention heads.


### RoBERTa
RoBERTa removes the Next Sentence Prediction (NSP) objective from BERT as well as training the model on longer sequences with much larger batch sizes. The masked language modelling objective showed an improved perplexity while improving the prediction accuracy. An additional feature which further improved the model’s accuracy was to dynamically change the masking pattern. RoBERTa was trained on data equal to 160 GB while using byte level byte-pair encoding for tokenization which removes the model’s reliance on unknown tokens.
