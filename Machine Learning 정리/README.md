## 머신러닝에 필요한 수학적 개념
##### 공분산 : x,y가 동시에 변하는 정도를 공분산이라 한다. https://blog.naver.com/xodh16/220545875799

## TensorFlow의 동작 원리
#### 연산은 graph로 표현한다. graph는 node와 edge로 이루어진 구조를 의미
#### graph는 Session 내에서 실행됨
#### 데이터는 tensor(다차원 배열)로 표현됨

## TensorFlow란?
#### TensorFlow는 graph로 연산을 나타내는 프로그래밍 시스템
#### TensorFlow 프로그램은 graph를 조립하는 구성단계(construction phase)와 session을 이용해 graph의 op를 실행시키는 실행단계(execution phase)로 구성된다.

##### TensorFlow를 쓰는 이유? <출처 : https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/mnist/pros/>
##### 파이썬에서 효율적인 수치 연산을 하기 위해, 우리는 다른 언어로 구현된 보다 효율이 높은 코드를 사용하여 행렬곱 같은 무거운 연산을 수행하는 NumPy등의 라이브러리를 자주 사용합니다. 그러나 아쉽게도, 매 연산마다 파이썬으로 다시 돌아오는 과정에서 여전히 많은 오버헤드가 발생할 수 있습니다. 이러한 오버헤드는 GPU에서 연산을 하거나 분산 처리 환경같은, 데이터 전송에 큰 비용이 발생할 수 있는 상황에서 특히 문제가 될 수 있습니다.
##### 텐서플로우 역시 파이썬 외부에서 무거운 작업들을 수행하지만, 텐서플로우는 이런 오버헤드를 피하기 위해 한 단계 더 나아간 방식을 활용합니다. 파이썬에서 하나의 무거운 작업을 독립적으로 실행하는 대신, 텐서플로우는 서로 상호작용하는 연산간의 그래프를 유저가 기술하도록 하고, 그 연산 모두가 파이썬 밖에서 동작합니다 (이러한 접근 방법은 다른 몇몇 머신러닝 라이브러리에서 볼 수 있습니다).
##### TensorFlow는 계산을 위해 고효율의 C++ 백엔드(backend)를 사용합니다. 이 백엔드와의 연결을 위해 TensorFlow는 세션(session)을 사용합니다. 일반적으로 TensorFlow 프로그램은 먼저 그래프를 구성하고, 그 이후 그래프를 세션을 통해 실행하는 방식을 따릅니다. 
##### InteractiveSession 클래스를 사용하는 이유 - TensorFlow 코드를 보다 유연하게 작성할 수 있게 해 주는 InteractiveSession 클래스를 사용할 것입니다. 이 클래스는 계산 그래프(computation graph)를 구성하는 작업과 그 그래프를 실행하는 작업을 분리시켜 줍니다. 즉, InteractiveSession을 쓰지 않는다면, 세션을 시작하여 그래프를 실행하기 전에 이미 전체 계산 그래프가 구성되어 있어야 하는 것입니다.

##### 과적합을 막기 위한 3가지 방법 1. Regularization - 특정 가중치가 큰 값을 갖 되는 것을 막는다. 2. 지능적 훈련 데이터 생성 - 기존의 데이터를 변형시켜 데이터를 얻는다. 3. Dropout - 입력 layer나 일부 뉴런을 생략한다.

##### Drop out 했을 때의 효과 1. Voting 효과 - 일정한 미니배치동안 줄어든 망을 이용해 학습을 하게 되면 그 망은 그 망 나름대로 overfitting이 되고 다른 미니배치 구간 동안 다른 망에 대해 학습을 하게 되면 그 망에 대해 다시 일정 정도 overfitting이 된다. 이런 과정을 무작위로 반복을 하게 되면, voting에 의한 평균 효과를 얻을 수 있기 때문에 결과적으로 regularization과 비슷한 효과를 얻을 수 있게 된다. 2. Co-adaptation을 피하는 효과 - 특정 뉴런의 바이어스나 가중치가 큰 값을 갖게 되면 그것의 영향이 커지면서 다른 뉴런들의 학습 속도가 느려지거나 학습이 제대로 진행이 되지 못하는 경우가 있다. 하지만 dropout을 하면서 학습을 하게 되면 결과적으로 어떤 뉴런의 가중치나 바이어스가 특정 뉴런의 영향을 받지 않기 때문에 결과적으로 뉴런들이 서로 동조화(co-adaptation)이 되는 것을 피할 수 있다. 특정 학습 데이터나 자료에 영향을 받지 않는 보다 강건한(robust)한 망을 구성할 수가 있게 되는 것이다. 이 논문 읽어보자 Dropout : A Simple Way to Prevent Neural Networks from Overfitting

##### Hyperparameter에는 어떠한 것들이 있을까?
##### 1. Learning Rate 2. Cost Function 3. Regularization parameter 4. Mini-batch 크기 5. Training 반복 횟수 6. Hidden unit의 개수 7. 가중치 초기값 - 보통 뉴런의 개수 제곱의 역수 범위 내에서 무작위로 설정해 줌 8. CNN에서 필터의 크기 9. Stride 10. Bias


##### Batch Normalization에 대한 글 (출처 - http://blog.naver.com/PostView.nhn?blogId=laonple&logNo=220808903260&parentCategoryNo=&categoryNo=21&viewDate=&isShowPopularPosts=true&from=search)
##### 딥러닝에서는 기울기 소실(vanishing/exploding gradient)가 가장 골칫거리중 하나인데 layer 수가 적으면 문제가 되지 않지만 layer 수가 많으면 기울기 소실이 나올 확률이 높다. 그 이유는 활성 함수로 sigmoid나 tanh와 같은 non-linear saturating function을 사용하기 때문인데 입력의 절대값이 작은 일부 구간을 제외하면 미분값이 0 근처로 가기 때문에 역전파를 통한 학습이 어려워지거나 느려지게 된다. 이 문제에 대한 해결책으로 2011년 ReLU(Rectifier Linear Unit)을 활성함수로 쓰는 방법이 소개되었지만 이것은 간접적인 회피이지 본질적인 해결책이 아니기 때문에 망이 깊어지면 여전히 문제가 된다. 그러다가 2015년에 Batch Normalization과 Residual Network가 소개되었고 획기적으로 여겨졌다. Batch normalization에서 알아야 할 전제 조건은 망이 깊어짐에 따라 이전 layer에서의 작은 파라미터 변화가 증폭되어 뒷단에 큰 영향을 끼치게 된다는 것이다. 이처럼 학습하는 도중이 이전 layer의 파라미터 변화로 인해 현재 layer의 입력의 분포가 바뀌는 현상을 Covariate Shift라고 한다. Covariate Shift를 줄이는 방법은 각 layer로 들어가는 입력을 whitening 시키는 것인데 이는 입력을 평균 0, 분산 1로 바꿔준다는 것이다.

##### 이렇게 하면 실행이 안된다.
```python
def prac(a, b):
        sess = tf.InteractiveSession()
        
        W = tf.get_variable("W", [a, b], initializer=tf.random_normal_initializer(stddev=0.1))
        sess.run(tf.global_variables_initializer())
        return W
    
    def inter():
        sess = tf.InteractiveSession()
        
        c = prac(3, 5)
        d = prac(3, 5) #두번 사용할 수가 없다.
        print(c.eval())
        print(d.eval())
```
##### 이렇게 에러가 뜬다. Variable W already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at:

##### get_variable을 쓰지 않으면 일일이 변수를 다 만들어줘야하는 불편함이 있다. 은닉층 한두개면 상관 없겠지만 많아지면 일일이 변수를 다 써야하므로 불편하다.

```
ResourceExhaustedError (see above for traceback): OOM when allocating tensor with shape[10000,32,28,28] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
	 [[Node: Conv2D = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], padding="SAME", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true, _device="/job:localhost/replica:0/task:0/device:GPU:0"](Reshape, Variable/read)]]
Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom to RunOptions for current allocation info.
```

## 주성분 분석 - 데이터 집합 내에 존재하는 각 데이터의 차이를 가장 잘 나타내 주는 요소를 찾아 내는 방법
##### http://adnoctum.tistory.com/977 참고하라
##### 개별데이터가 n개, 변수가 p개 있다고 하자
```
{X={x11, x21, ...., xp1},
 {x21, x22, ...., xp2},
 ...
 ...
 {xn1, xn2, ...., xpn}}

 W={w1, w2, ...., wp}

 T = X*W(t)
```

##### X*W(t)(내적)을 구했을때 결과는 T={t1, t2, ...., tn}이 나오게 되는데 이 결과(T)의 분산이 최대값이 나오도록 하는 W를 찾는 것이 핵심

##### 수식으로 살펴보면 var(T) = var(t1, t2, t3, ...., tn)이고
##### W = arg max Var(t1)이다. 그리고 ||W||는 1인데 이것은 W의 총 크기가 1이라는 이야기다. (w1^2 + w2^2 + w3^2 + .... + wp^2)^(1/2)=1

##### 분산의 공식이 V(Y) = E(Y^2) - [E(Y)]^2 인데 E(Y)를 0으로 되게 데이터를 만들면
##### V(Y)는 E(Y^2)가 되고 결국 W = arg max Var(T) = arg max E(T^2)이다
 
##### 결론부터 적으면 W=arg max{w(t)X(t)Xw/w(t)w}에서 X(t)X의 eigenvalue의 최대값이 되고 W는 그 고유값의 고유벡터가 된다. 제2주성분의 값은 원래 X에서 제1주성분을 빼고 위의 과정을 거치는 것이지만 X(t)X의 두번째 크기의 eigenValue값에 해당하는 eigenVector가 된다. 중간 과정은 더 공부를 해보자(에르미트 행렬 등등)





