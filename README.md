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


-------------------------------------------------------------------
1. 모델을 적용하기전, 데이터를 살펴보지 않아, id를 왜 변경해줘야 하는지 이해하지 못함. 
- 분석전에 데이터파일 먼저 열어서 어떻게 전처리해야 모델에 적용할 수 있는지를 반드시 생각해봐야 한다. 
- ( feature의 의미 뿐만 아니라 데이터 자체가 어떻게 구성되어 있는지를 봐야 한다. ) 

2. cnn 모델에 대한 이해가 필요했음. 이 문제에서는 multi regression을 사용하는 것보다 
- cnn을 사용하는 것이 더 좋은 결과를 얻을 수 있었음. 따라서 resnet이나 vggnet 모델을 생각해봄. 
- **근데 여기서 왜 cnn을 사용해야 하는지 궁금함.(결정하는 이유나 방법에 대해)**
- cnn은 이미지 데이터에서만 쓰는 방법은 아님. 2016-2017 연구에서 가장 예측력이 좋은 모델이었음
- 즉 이미지 데이터라는 것은 결국 2차원 이상의 matrix를 의미한다. 
- 데이터를 2차원 이상의 매트릭스로 정리하는 것은 생각보다 간단함
- 따라서 이미지 데이터가 아닌 데이터를 어떻게든 2차원 매트릭스로 변환하여 cnn을 적용해보고자 하는 연구가 있었다. 
- 자연어 인식에서 다루기도 하는 방법이다. 

3. 데이터 전처리 방법
- 신호데이터와 같은 경우는 푸리에를 적용하여 전처리도 가능했다는 점. 

4. 이슈1에 대한 해결
- X,Y,M,V를 다같이 하나의 모델로 학습시키는 것보다 각각 따로 학습시키는 것이 더 좋은 결과를 얻을 수가 있었다. 
- ( 더 정확히 말하면 위치 target과 질량 target, 속도 target으로 나누어 학습시킴 )

5. **data agumentation (데이터 증식)할 수 있는 방법?(이슈2를 사용할 수 있는 방법)**
- 혹은 시계열 데이터 그대로 사용할 수도 있지만 주파수 domain을 변형하여 사용할 수 있는 방법?
--> 이 방법은 어떻게 응용해 볼 수 있는지가 궁금함.