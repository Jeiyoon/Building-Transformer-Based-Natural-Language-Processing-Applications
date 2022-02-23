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

## 혜택: 

~~~
- 엠베서더 등급제: 엠베서더 취득 후 여러 활동을 하면 등급이 올라감
- 비용지원 연구프로그램 우선선발
- 연구협력 (TBA)
- 산학 협력
- HR혜택 (TBA)
- 입사 할때 엠베서더는 추가 혜택
~~~

## 예상 질문:

~~~
- 트랜스포머 이전의 신경망 이전에 중요한 아키텍쳐가 뭐뭐 있냐
- NLP처리과정(토크나이징 하는 과정 왜 토크나이징을 진행했는지 등 기존과정의 문제점과 이를 메이크업하기위한 방법론)
- 실제 트렌스포머의 장점이 뭔지
- 트랜스포머의 대표적인 아키텍쳐의 차이점들
- 트랜스포머류 모델 발전과정의 두가지 흐름 (데이터셋의 흐름, 모델이 커짐)
- 로버타 or XLN 같은건 안중요함 (그래도 일단은 공부)
- 버트 GPT제대로 알아라
- NLP에대한 윤리적인 문제
~~~

## you must be familiar with, and prepared to explain, the following concepts or terminology:

| NLP tasks |  NLP challenges  |
| :---: | :---: |
| **Softmax classifiers** |  **SGD/GD**  |
| **Cross-entropy** | **CNNs** |
| **RNNs/LSTMs** | **GANs** |
| **AutoEncoders** | [**Reinforcement Learning**](https://jeiyoon.github.io/data/ipa_3.pdf) |
| **Natural Language Processing (basics)** | **Training, validation, and test datasets** |
| **Pooling** | **Padding** |
| **Overfitting** | **Hyperparameters** |
| **Features** | **Parameters** |
| **Activation functions** | **Transfer Learning** |
| **Frameworks** | **Image Classification** |
| **Dropout** | **Forwards and backwards propagation** |


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
