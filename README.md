## 진동데이터 활용 충돌체 감지 

- 원자력발전소 냉각재계통 내부에서 충돌체가 충돌했을 때의 충격파를 감지하여 충돌했을 때의 위치, 질량, 속도를 예측
- 원전 현장에서는 기기의 이상징후를 조기에 진단하여 사고를 방지하고자 함. 
- 주파수 특성, 타임 도메인 등을 활용하여 정확히 예측해보고자 한다. 
- 즉 시간/가속도 데이터를 바탕으로 역으로 충돌물의 위치를 파악하고자 하는 것이 목표. 

- 시간차이(가속도)를 추정하여 해결한다. 
- 4개의 가속도 센서의 축에 따른 가속도 속도의 변화를 감지하여 2800개의 경우에 대하여 학습시킴
- 평가기준 : E1 -> 거리오차, E2-> 질량과 속도의 상대오차
-----------------------------------------------------------------
#### issue 

- **이슈1** : 단일모델 사용 vs 다중모델 사용
- **이슈2** : 시계열 데이터를 그대로 볼 수도 있지만, 주파수 도메인 변경 혹은 data augumentation을 사용해 볼 수도 있겠다는 점. 
------------------------------------------------------------------
#### code file 

(1) **시도1.ipynb**
- data를 확인해본 결과, id(충돌체별 고윳값)별로 묶어 특성을 구분할 필요가 있어, time(관측시간)과 각 센서별 가중치를 합하여 column으로 지정함
- randomforest 앙상블 모델을 사용하고 각각을 randomsearch를 사용하여 하이퍼파라미터 튜닝완료 

(2) **시도2.ipynb**
- 이번엔 cnn모델에 적용해보고자 함. (시도3에서는 resnet이나 vggnet을 사용해 보고자 했다) 
- 먼저 데이터를 .reshape((2800, 375, 5, 1))의 형태로 변형함. 
- 각 id별로 375개의 데이터가 2800개 있는 형태이므로 위와 같이 변형함. 
- 노드수를 16부터 시작하여 2배씩 늘려가며 적용하고, 마지막엔 완전 연결층으로 연결
- Adam optimizer을 사용하고 batchsize는 256으로 지정함,
- 위치(x,y좌표), 질량(m), v(속도)를 따로 학습시켜 예측하는 방법을 사용함. 

(3) **시도3. ipynb**
- apply resnet, vggnet algorithm 
- data augmentation(using time frequency domain)

![image](https://user-images.githubusercontent.com/49298791/86868357-81870080-c10f-11ea-9c53-654e2f24ae36.png)

-------------------------------------------------------------------
#### what I do 

#### 1. 모델을 적용하기전, 데이터를 살펴보지 않아, id를 왜 변경해줘야 하는지 이해하지 못함. 
- 분석전에 데이터파일 먼저 열어서 어떻게 전처리해야 모델에 적용할 수 있는지를 반드시 생각해봐야 한다. 
- ( feature의 의미 뿐만 아니라 데이터 자체가 어떻게 구성되어 있는지를 봐야 한다. ) 


#### 2. cnn 모델에 대한 이해가 필요했음. 이 문제에서는 multi regression을 사용하는 것보다 
- cnn을 사용하는 것이 더 좋은 결과를 얻을 수 있었음. 따라서 resnet이나 vggnet 모델을 생각해봄. 
- **근데 여기서 왜 cnn을 사용해야 하는지 궁금함.(결정하는 이유나 방법에 대해)**
- cnn은 이미지 데이터에서만 쓰는 방법은 아님. 2016-2017 연구에서 가장 예측력이 좋은 모델이었음
- 즉 이미지 데이터라는 것은 결국 2차원 이상의 matrix를 의미한다. 
- 데이터를 2차원 이상의 매트릭스로 정리하는 것은 생각보다 간단함
- 따라서 이미지 데이터가 아닌 데이터를 어떻게든 2차원 매트릭스로 변환하여 cnn을 적용해보고자 하는 연구가 있었다. 
- 자연어 인식에서 다루기도 하는 방법이다. 
- rnn이나 cnn은 등장 순서가 중요한 sequential data를 처리하는데 강점을 가진다. 
cnn은 필터가 움직이면서 지역적인 정보를 추출, 보존하는 형태로 학습이 이뤄진다. 
요컨데 RNN은 단어 입력값을 순서대로 처리함으로써, CNN은 문장의 지역정보를 보존
함으로써 단어/표현의 등장순서를 학습에 반영하는 아키텍쳐 이다. 
cnn에서는 필터를 지정하고 필터 개수만큼의 feature map을 만들고 max-pooling의 과정을 거쳐
스코어를 출력하는 네트워크 구조이다. 
자연어처리에서는 단어벡터를 랜덤하게 초기화한 후 학습과정에서 이를 업데이트하면서
쓰는 방법을 채택한다. 이런 방식을 사용하기 위해서는 텍스트 문장을 나열로 변환해야 한다
(참고 블로그 ㅣ https://ratsgo.github.io/natural%20language%20processing/2017/03/19/CNN/)


#### 3. 데이터 전처리 방법
- 신호데이터와 같은 경우는 푸리에를 적용하여 전처리도 가능했다는 점. 

#### 4. 이슈1에 대한 해결
- X,Y,M,V를 다같이 하나의 모델로 학습시키는 것보다 각각 따로 학습시키는 것이 더 좋은 결과를 얻을 수가 있었다. 
- ( 더 정확히 말하면 위치 target과 질량 target, 속도 target으로 나누어 학습시킴 )


#### 5. **data agumentation (데이터 증식)할 수 있는 방법?(이슈2를 사용할 수 있는 방법)**
- 혹은 시계열 데이터 그대로 사용할 수도 있지만 주파수 domain을 변형하여 사용할 수 있는 방법?
**(1) use pooled design or ensemble model** 
**(2) new method for signal data**
  - ***1) signal segmentation and recombination in the time domain.***
	- 동일한 class에서 데이터를 segmentation하고 random하게 선택하여 concat함으로서 artifitial한 새로운 데이터를 생성한다
	- 동일 class내에서 하는 작업이므로 feature를 해치치 않는 관점에서 유용한 방법이다. 
	- 실제로 covariance matrix를 통해 LDA classifier를 사용한 것과 다를 바 없음을 보일 수 있다. 
  - ***2) signal segmentation and recombination in the time frequency design***
	- 앞의 방법대로 한다면 단순히 segment값을 concat하는 방식이므로 원치 않는 noise가 발생하게 된다. 
	- 이를 해결하기 위해 "time frequency domain"을 사용한다. 
	- transform each bond-passed filtered training trial Ti in a time-frequenct representation TFi using STFT.
	- TFI_k는 결국 kth time window를 의미한다.
	- 즉 concatenating together using STFT windows를 하면서 새로운 artifitial data를 생성한다. 

![image](https://user-images.githubusercontent.com/49298791/86796057-27535480-c0a9-11ea-95c8-fdf114146765.png)


  - ***3) aritifitual trial genertion based on analogy.***
	- "computing transformation to make trial a similar to trial B and then applying this transformation to trial C and create
	  artifitial trial D"
	- 먼저 각 class 의 available data에 대하여 covariance matrix C를 구한다. 
	- 이를 바탕으로 고유벡터 V를 구한다 (Princopal Component in data)
	- and randomly select 3 distinct trials Xa, Xb, Xc
	- project first two of them to the PC of data and compute signal power pa_i and pb_i.
	- make Xd using Xc 
(참고 논문 | Signal processing approaches to minimize or suppress calibration time in oscillatory activity-based Brain-Computer Interfaces )



#### 6. **keras.application.ResNet50 framework가 아니라 function을 직접 작성한 이유**
- residual net : using shortcut and skip connection allows the gradient be
directly backpropagated to earlier layers.
- the identitiy block is the standard block used in ResNets, and corresponds to the
case where the input activation has the same dimention as the output activation
- the convolutional block is the other type of block. I can use this type when the
input and output dimentions don't match up.
- why do skip connections work?
1) they mitigate the problem of vanishing gradient by allowing this alternate
shortcut path for gradient to flow throught
2) they allow the model to learn an identity function which ensures that
the higher layer will perform at least as good as the lower layer, and not worse.
- cnn은 학습완료함. resnet은 안되는 이유?
- resnet에서 weight와 train_target이 꼬여 있어서 문제 발생 
- 원래 keras.application,resNet50을 사용하면 각각 학습이 불가능하다.
- **따라서 resNet50 library를 뜯어서 각각 학습이 가능하도록 만들었음.**
- 바꾼 부분은 해당하는 weight의 경우, loss를 위치 / 질량과 속도에 따라 다르게 구하므로
이를 각각 weight를 지정하면서 지정해 주었다. 
- 또한 ensemble design로 만든 형태이므로, 각 weight를 원래 라이브러리에서 
지정하는 형태가 아닌 다른 형태로 가져와 지정해 줬다. 
