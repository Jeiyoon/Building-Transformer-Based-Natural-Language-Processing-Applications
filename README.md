## Questions:

~~~
1 .

"What was the important neural network architecture yard before Transformer came out?"

- Multilayer perceptron (MLPs), CNNs, and RNNs
- CNNs are useful neural network which is used for finding and analyzing patterns from images or videos.
- RNNs are sequence models that processes inputs and outputs in sequence units.
~~~

~~~
2. 

"What is tokenization?"

- Given a corpus, dividing this corpus into tokens is called tokenization.
- A token is usually defined as a meaningful unit such as word, sentence, and morpheme.


"Describe the process of tokenizing."

- Basically previous tokenization methods consist of word tokenization, sentence tokenization, morpheme tokenization and so on.
- And the tokenization rules should be defined according to the language, or how special characters are used within the corpus.
- A sentence tokenization, for instance, can be divided based on punctuation. 
- But, punctuation or special characters should not be simply excluded, but rather separated according to more complex rules.


"Why do we need tokenization?"

- The reason for tokenization is to make the language recognizable and able to be processed by the computer.


"Please explain the problems of the existing tokenization methods and how to solve them."

- No matter how many words we teach our computer, we can't let a computer know all the words in the world.
- If a word that the computer does not know appears, the token is expressed as UNK (Unknown Token) 
in the sense that the word is not in the word set.
- This out-of-vocabulary (OOV) problem sharply degrades the performance of natural language processing models.
- So we can employ Subword segmenation method such as Byte pair encoding (BPE) to encode and embed a word
by splitting it into multiple subwords.

"Upside and downside of Subword tokenizier"

- We can get more tokens from a single word.
- 
~~~

~~~
3. 

"Please explain the advantages of transformers compared to existing neural networks."
- Previous most of the deep learning models based on Recurrent Neural Network (RNN) or RNN + Attention are used.
- However, RNN-based deep learning models have a problem of sequentially processing input sequence data 
rather than parallel processing.
- Also, there is a problem of long-term dependency according to the length of the input sequence.
- Therefore, the Transformer, which overcomes the shortcomings of RNNs by only the attention technique 
without using the RNN things, has a great advantage compared to the existing models.
~~~

~~~
4. 

"What are typical models using Transformer?"

(1) BERT
- Transformer encoders are stacked in both directions.
- BERT is pretrained on a large amount of data, and then fine-tuned for downstream tasks.

- BERT has the advantage of not having to change the model design for each task.
- Pretrain process consists of "Masking language model" and "Next sentence prediction".
- "Masked language model" learns by applying a mask randomly in three ways with 
a 15% probability and matching it 
- Given two sampled sentences, "Next sentence prediction" learns by changing following sentence 
with a 50% probability, and guessing whether it has changed.

- However, AE is not without its drawbacks.
- The biggest problem is that it assumes that the masked tokens are independent of each other.
- In this case, the dependency between the masking tokens cannot be determined.

- BERT also has the disadvantage that it is difficult to learn long contexts.


(2) GPT-n

<GPT>
- GPT is a forward language model using a Transformer decoder stacks.
- Unsupervised pre-training using unlabeled text
- and then Supervised fine-tuning using manually annotated dataset


<GPT-2>
- Zero-shot task transfer, When only minimal or no supervised data is available, and
Ability to optimize the unsupervised objective to convergence
- Auto-regressive model
- Masked Self-Attention


<GPT-3>
- Few-shot learning
- Task prompt


(3) Ernie
- ERNIE(Enhanced Representation through kNowledge IntEgration)

- Most of the previous models just predict the masked word within the context, 
which does not take into account prior knowledge of the sentence.
- For example, in the sentence “Harry Potter is a series of fantasy novels written by J. K. Rowling”, 
Harry Potter is the name of the novel, J.K. Rowling is the author's name.
- Unlike human, the model can't recognize Harry Potter and J.K. Rowling's relationship.

- BERT masks each word with a certain probability and predicts it, 
but ERNIE masks not only words but also entities and phrases as a unit.

- Transformer Encoder

- Phrase-level masking is a conceptual unit that masks an entire phrase composed of several words.
- Entity-level masking masks entities consisting of multiple words.


(4) XLNet
- Alleviating BERT's problems
- To overcome the limitations of AR and AE models, XLNet proposed a permutaion language model.
- Unlike BERT, which assumes independence between predicted words (masking tokens), 
the permutation language model is advantageous for capturing inter-word dependencies.

- Two-Stream Self Attention: query stream and content stream
- When creating a content stream, it utilizes the token information corresponding to the previous context and itself.
- Until now, I've seen the words "school", "yesterday", "go to", The word to match this time was first in the original sentence.
What is this word?

- Due to permutation, the model faces a contradiction in that it takes the same input and produces different outputs.

~~~


~~~
5.
What is the problem with bag of words?
- Bag of Words(BOW) is a model that creates feature values by ignoring the context or order of words
and assigning frequency values to words.

- Since BOW does not consider the order of words, the contextual meaning is ignored.
- To alleviate this, the n_gram technique can be used, but it is limited.

- Words are far more likely to not appear from document to document because each document is made up of different words.
- Therefore, most columns will be filled with zeros.
- In a matrix composed of a large number of columns, a matrix in which most values are filled with zeros is called a "sparse matrix".
~~~

~~~
6.
"Word2vec vs. Glove"

- Word2Vec  
- There are two categories of methods in Word2vec: CBOW (Continuous Bag of Words) Skip-Gram. 
- CBOW generates word vectors by matching the central word with the surrounding words.
- Skip-Gram predictㄴ which words will come before and after a key word.
- Word2Vec locally calculates cosine similarities given a sentence.

- Glove (Global Vectors)
- Authors criticized that Word2Vec poorly utilizes the statistics of the corpus since they train on separate local context windows instead of on global co-occurrence counts.
- Glove aims to consider the statistical information of the corpus while it measures the similarity between word vectors using "the words’ probability of co-occurrence".
~~~

~~~
7.

"Transformer details"

(1) No RNNs and Convolution

- The Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.

(2) Why normalization?

- In NLP tasks, the sentence length often varies. Thus, if using batchnorm, it would be uncertain what would be the appropriate normalization constant because of the total number of elements to divide by during normalization to use.
- It can be proved that at the beginning of the optimization, for the original Transformer, which places the layer normalization between the residual blocks, the expected gradients of the parameters near the output layer are large. Then using a large learning rate on those gradients makes the training unstable. The warm-up stage is practically helpful to avoid this problem. 
- Such an analysis motivates us to investigate a slightly modified Transformer architecture which locates the layer normalization inside the residual blocks

(3) Why Transformer employs encoder-decoder architecture? (why sharing matrix)

- The encoder maps an input sequence of symbol representations to a sequence of continuous representations.
- The decoder then generates an output sequence of symbols one element at a time.

(4) Why multi-head attention? not single head?

- It allows the model to focus on different postions or sub-spaces

(5) Positional Encoding

- Authors hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of PEpos.
- It prevents the distance between two words from becoming too small.
- It equalizes the distance between two words.

(6) Drawbacks

- Since a segment of a fixed length is used as an input, it is difficult to learn a dependency longer than the corresponding length.
- "Context Fragmentation": Since each segment was used by simply cutting consecutive tokens without considering semantic boundaries (sentences, etc.), the amount of information is insufficient to predict the first few tokens of the segment. 
~~~

~~~
8.

"Two upsides of TransformerXL"

- Segment-level recurrene mechanism (like RNN)
- Relative positional encoding scheme (transformer -> absolute positional encoding): Instead of an absolute encoding representing the position of each token, it creates a relative encoding between 0 and Lmax representing the distance between two tokens.
~~~

~~~
9.

"A downside of BERT"

- The biggest problem is that it assumes that the masked tokens are independent of each other.
- In this case, the dependency between the masking tokens cannot be determined.
- BERT also has the disadvantage that it is difficult to learn long contexts.
- These are the reasons why "permutation LM (e.g. XLNet)" is needed
~~~

~~~
10.

"NVIDIA Products"

(1) NeMo

- NVIDIA NeMo is an interactive AI toolkit for researchers in automatic speech recognition (ASR), natural language processing (NLP), and speech synthesis systems (TTS).
- NeMo consists of pytorchlightning & config file (yaml): traning script (three lines)

(2) Hydra-runner Decorator (@hydra_runner)

- Hydra_runner links configuration files and provides a mechanism for redefining the command line.
- Simplify training process 
- Let us know where configuration at 

(3) Omegaconf

- Each YAML section can be checked more easily with the omegaconf package

(4) NVIDIA Triton Inference Server 

- Simultaneous model execution (multiple models can be run simultaneously)
- Dynamic batching (higher throughput)
- Model live replacement (can be updated while server is running)
- Docker container available (portable)
- Multiple framework support (TensorRT, TensorFlow, PyTorch, ONNX)
~~~

~~~
et cetra.

- 트랜스포머류 모델 발전과정의 두가지 흐름 (데이터셋의 흐름, 모델이 커짐)
- NLP에대한 윤리적인 문제
- nvidia 제품 (e.g. megatron, 쿠버네티스)
- BERT: 30K
- one-hot -> BERT input (그래서 보캡사이즈가 30K임)
- 768 -> length of QKV (transformer는 512)
- 512 -> positional encoding max sequence length
- 2 -> sentence A and sentence B
~~~

## You must be familiar with, and prepared to explain, the following concepts or terminology:

- The format for demonstrating knowledge in these areas will be a question and answer session
with the expectation that you can demonstrate the ability to field student questions and
understand the concepts at a deeper level than those attending a workshop.

| NLP tasks |  NLP challenges  |
| :---: | :---: |
| **Embeddings** |  **Word2Vec and Glove**  |
| **LSA / LDA** | **Auto Encoder** |
| **RNNs / LSTMs** | **ELMO** |
| **ULM-FIT** | **NMT**|
| **Seq2Seq** | **Positional encoding** |
| **Self-attention** | **Multi-head attention** |
| **Masking** | **GPT, GPT2, GPT3** |
| **BERT** | **RoBERTA** |
| **Megatron** | **Transformer XL** |
| **XLNet** | **ALBERT** |
| **ELECTRA** | **ERNIE 2.0** |
| **T-NLG** | **NeMo** |
| **BioBERT** | **BioMegatron** |
| **TensorRT** | **NVIDIA Triton Inference Server** |
| **Kubeflow** | **NVIDIA Riva** |
| **ONNYX** | **BLEU** |


- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest 
Can Be Pruned](https://arxiv.org/pdf/1905.09418.pdf) 
- [The Evolved Transformer](https://arxiv.org/pdf/1901.11117.pdf)
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) 
- [Stanford CS224N: NLP with Deep Learning – Winter 2019 – Lecture 14 – Transformers
and Self-Attention](https://www.youtube.com/watch?v=5vcj8kSwBCY)
- [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 8 – Translation,Seq2Seq, Attention](https://www.youtube.com/watch?v=XXtpJxZBa2c&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=8)
  - Includes a detailed discussion on Beam Search, BLEU and NMTs
- [Quick overview of the concept of embeddings](https://www.youtube.com/watch?v=186HUTBQnpY)
- [Learning embeddings in an embedding layer](https://www.youtube.com/watch?v=xtPXjvwCt64)
- [Embeddings developed using GloVe](https://www.youtube.com/watch?v=oUpuABKoElw)
- [NLP Tasks Leaderboard](https://www.paperswithcode.com/area/natural-language-processing)
- [NLP Datasets](https://machinelearningmastery.com/datasets-natural-language-processing/)
  - [Contains SOTA for common NLP tasks and datasets](https://nlpprogress.com/)
- [GLUE Benchmark](https://gluebenchmark.com/)
- [positional encoding](https://skyjwoo.tistory.com/entry/positional-encoding%EC%9D%B4%EB%9E%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80)


## Memo

~~~
chapter 1

- LSA
- LDA
- Auto-encoder
- Word2vec
- Glove
- problems of RNNs
- Why attention (not transformer and transformer)
- transformer is faster than RNNs

  - RNN의 단점: 한 번에 한 단어를 읽어야 함
  - 그래서 한번에 한 단어를 읽어야 하므로 결과를 얻으려면 여러 단계를 수행해야함
  - 순차적인 처리 -> 병렬화를 못해서 느림
  - 트랜스포머는 그에 비해 병렬화에 효율적
  - 즉, (1) 트랜스포머는 반복을 Attention으로 대체하고 입력 시퀀스 내에서 각 기호의 위치를 인코딩하여 병렬화함. 
트레이닝 시간 단축
  - (2) 트랜스포머는 입력/출력 시퀀스의 두 기호를 O(1) 연산 수에 연결하여 순차 연산의 수를 줄임.
트랜스포머는 입력 또는 출력 문장에서의 거리에 관계없이 종석성을 모델링할 수 있게 해주는 Attention 매커니즘을 사용하여 이를 구현

- ELMO
- UML-FIT
~~~
