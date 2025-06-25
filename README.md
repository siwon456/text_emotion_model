# text_emotion_model

감성 분석을 위한 BiLSTM 모델 기술 보고서
# 1. 개요 (Overview)
본 보고서는 게임 리뷰 텍스트의 감성을 '부정', '중립', '긍정'의 세 가지 클래스로 분류하기 위해 설계된 딥러닝 모델에 대해 기술합니다. 모델은 자연어 처리에서 뛰어난 성능을 보이는 양방향 LSTM (Bidirectional Long Short-Term Memory, BiLSTM) 아키텍처를 기반으로 하며, 과적합을 방지하고 일반화 성능을 높이기 위한 여러 기법이 적용되었습니다.

# 2. 모델 아키텍처 (Model Architecture)
본 모델은 Keras Sequential API를 사용하여 구축되었으며, 입력부터 출력까지 총 6개의 레이어로 구성됩니다.

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding (Embedding)       (None, 147, 64)           ...

 bidirectional (Bidirectional) (None, 128)               66048

 dropout (Dropout)           (None, 128)               0

 dense (Dense)               (None, 32)                4128

 dropout_1 (Dropout)         (None, 32)                0

 dense_1 (Dense)             (None, 3)                 99
=================================================================
# 2.1. 임베딩 레이어 (Embedding Layer)
역할: 텍스트의 각 단어를 나타내는 정수 인덱스를 고밀도의 실수 벡터(Dense Vector)로 변환합니다. 이 과정을 통해 단어 간의 의미적, 문법적 관계를 벡터 공간에 표현할 수 있습니다.

주요 파라미터:

input_dim (vocab_size): 모델이 인식하는 총 단어의 수 + 1

output_dim: 64. 각 단어가 표현될 임베딩 벡터의 차원입니다.

input_length: 147. 모델에 입력되는 문장의 최대 길이입니다.

# 2.2. 양방향 LSTM 레이어 (Bidirectional LSTM Layer)
역할: 문장의 맥락을 양방향으로 학습합니다. LSTM은 순환 신경망(RNN)의 한 종류로, 문장의 장기 의존성(Long-term Dependency)을 효과적으로 학습할 수 있습니다.

양방향(Bidirectional) 처리: 하나의 LSTM은 문장을 처음부터 끝으로(정방향), 다른 하나는 끝에서 처음으로(역방향) 처리합니다. 두 방향의 결과를 결합함으로써 각 단어의 의미를 예측할 때 앞뒤 문맥을 모두 고려할 수 있어 정확도가 향상됩니다.

주요 파라미터:

units: 64. LSTM 셀의 출력 차원입니다. 양방향으로 처리되므로 최종 출력은 이 값의 2배인 128차원이 됩니다.

# 2.3. 드롭아웃 레이어 (Dropout Layer)
역할: 훈련 과정에서 신경망의 과적합(Overfitting)을 방지하는 정규화(Regularization) 기법입니다.

작동 방식:

BiLSTM 레이어 다음에 Dropout(0.5)를 적용하여 LSTM 출력의 50%를 무작위로 비활성화합니다.

Dense 레이어 다음에 Dropout(0.4)를 적용하여 Dense 출력의 40%를 무작위로 비활성화합니다.

이를 통해 모델이 특정 뉴런에 과도하게 의존하는 것을 막고, 보다 강건한(Robust) 특징을 학습하도록 유도합니다.

# 2.4. 완전 연결 레이어 (Dense Layer)
역할: LSTM 레이어에서 추출된 고차원 특징을 조합하여 최종 분류에 사용될 특징을 만듭니다.

첫 번째 Dense 레이어:

units: 32. 32개의 뉴런을 가집니다.

activation: 'relu'. 비선형 활성화 함수 ReLU를 사용하여 모델의 표현력을 높입니다.

출력 Dense 레이어:

units: 3. 최종적으로 분류해야 할 클래스('부정', '중립', '긍정')의 수와 동일합니다.

activation: 'softmax'. 모델의 출력을 각 클래스에 대한 확률 값으로 변환합니다. 모든 출력의 합은 1이 되며, 가장 높은 확률 값을 가진 클래스를 모델의 최종 예측으로 사용합니다.

# 3. 훈련 프로세스 (Training Process)

# 3.1. 컴파일 설정
모델을 훈련시키기 위해 다음과 같은 설정을 사용했습니다.

Optimizer: Adam

학습률(Learning Rate)이 동적으로 조절되는 효율적인 최적화 알고리즘입니다.

learning_rate=0.0001: 비교적 낮은 학습률을 설정하여 안정적인 수렴을 유도합니다.

clipnorm=1.0: Gradient Clipping을 적용하여 그래디언트 폭주(Exploding Gradient) 문제를 방지하고 훈련 안정성을 높입니다.

Loss Function: sparse_categorical_crossentropy

다중 클래스 분류 문제에 사용되는 표준적인 손실 함수입니다. 특히, 타겟 레이블이 원-핫 인코딩(One-hot Encoding)이 아닌 정수 형태일 때 사용됩니다.

Metrics: ['accuracy']

훈련 및 검증 과정에서 모델의 성능을 측정하기 위한 지표로 '정확도'를 사용합니다.

# 3.2. 훈련 최적화
조기 종료 (Early Stopping):

모델의 과적합을 방지하고 최적의 훈련 시점을 찾기 위해 사용됩니다.

monitor='val_loss': 검증 데이터셋의 손실 값(val_loss)을 모니터링합니다.

patience=3: 3 에포크(Epoch) 동안 검증 손실 값이 개선되지 않으면 훈련을 자동으로 중단합니다.

restore_best_weights=True: 훈련이 중단되었을 때, 가장 낮은 검증 손실 값을 기록했던 시점의 모델 가중치로 복원합니다.
