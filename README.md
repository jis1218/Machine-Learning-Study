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