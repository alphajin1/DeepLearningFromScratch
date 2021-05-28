from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error


class Softmax:
    """
    구현 규칙
    1. 모든 계층은 forward()와 backward() method를 지닌다.
    2. 모든 계층은 인스턴스 변수인 params와 grads를 가진다.
        * params = weight, bias 같은 매개변수를 보관하는 리스트
        * grads = params 에 저장된 각 매개변수에 대응하여, 해당 매개변수의 기울기를 보관하는 리스트
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class MatMul:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.x = None

    def forward(self, x):
        w, = self.params
        out = np.dot(x, w)
        self.x = x
        return out

    def backward(self, dout):
        w, = self.params
        dx = np.dot(dout, w.T)
        dw = np.dot(self.x.T, dout)

        # 깊은복사 처리 (Numpy 배열이 가리키는 메모리 위치를 고정시킨 다음에, 그 위치에 원소들을 덮어씀)
        # 이렇게 하면, 기울기를 그룹화하는 작업을 최초에 한번만 하면 된다는 이점이 생긴다.
        self.grads[0][...] = dw
        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        # Sigmoid 미분값을 이용
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    # 완전연결계층에 의한 변환은 기하학에서 Affine 변환에 해당하기에 Affine 계층이라고 이름을 지음
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        # 헷갈릴 때는 이곳을 보자.
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        # 깊은복사 처리
        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:
    """
    Softmax + Loss
    Gradient = Vector의 각 원소에 대한 미분을 정리한 것이 기울기 (Gradient)

    """

    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax 의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        # Loss (Scala value)
        # L = f(x) (x = vector)
        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        """
        backward propagation
        Softmax + Loss 의 형태는 계산이 깔끔하게 떨어지니 참고할 것.
        """
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid의 출력
        self.t = None  # 정답 데이터

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Embedding:
    """
    word2vec 용
    단어 ID에 해당하는 행(Vector)를 추출하는 계층
    """

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0

        # For Loop Version
        # for i, word_id in enumerate(self.idx):
        #     dW[word_id] = dout[i]

        np.add.at(dW, self.idx, dout)
        return None
