## 특징: 

~~~
- 이번 NLP는 매우 어렵다 준비 많이해야된다. 면접 첫 질문만에 대답못해서 나온사람 많았다.
- NLP듣고 가속컴퓨팅이나 대화형 AI애플리케이션 따는거 추천
- 그리고 분기당 2개씩 업데이트된다네
- 소리님 인터뷰 들어간 것중에서 한분도 통과한사람이 없으시다네..
- 니모 & tryton 서비스, NLP 쪽에서의 큰 줄기 공부 필요, NLP히스토리에 대한 큰 줄기
- 면접 대답은 솔직히 그 슬라이드 만으론 부족
- 면접관 찰리 빡셈
~~~


## 예상 질문:

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
- 
~~~




~~~
- 
- NLP처리과정()
- 실제 트렌스포머의 장점이 뭔지
- 트랜스포머의 대표적인 아키텍쳐의 차이점들
- 트랜스포머류 모델 발전과정의 두가지 흐름 (데이터셋의 흐름, 모델이 커짐)
- 로버타 or XLN 같은건 안중요함 (그래도 일단은 공부)
- 버트 GPT제대로 알아라
- NLP에대한 윤리적인 문제
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


- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) [✔]
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf) [✔]
- [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest 
Can Be Pruned](https://arxiv.org/pdf/1905.09418.pdf) [✔]
  - takeaways
- [The Evolved Transformer](https://arxiv.org/pdf/1901.11117.pdf) [✔]
  - takeaways
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237.pdf) [✔]
  - takeaways
- [Stanford CS224N: NLP with Deep Learning – Winter 2019 – Lecture 14 – Transformers
and Self-Attention](https://www.youtube.com/watch?v=5vcj8kSwBCY)
  - takeaways
- [Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 8 – Translation,Seq2Seq, Attention](https://www.youtube.com/watch?v=XXtpJxZBa2c&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=8)
  - Includes a detailed discussion on Beam Search, BLEU and NMTs
  - takeaways

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
