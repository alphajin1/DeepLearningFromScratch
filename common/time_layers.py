# coding: utf-8
from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import sigmoid


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None  # back propagation 때의 사용됨.

    def forward(self, x, h_prev):
        # h_prev : 이전 RNN Output
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b  # b의 덧셈에서는 BroadCast 가 일어남.
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        return h_next

    def backward(self, dh_next):
        # 어차피, + 연산은 그냥 전파되는거라 생각보다 간단하게 끝이 난다.
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)  # dh_next * tanh 의 미분
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]

        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None # 다수의 RNN Layers 를 저장하는 용도

        # h : forward() 를 불렀을 때, 마지막 RNN 계층의 은닉 상태를 저장
        # dh : backward() 를 불렀을 때, 하나 앞 블록의 은닉 상태의 기울기를 저장
        self.h, self.dh = None, None
        # stateful : 상태가 있으면, 은닉 상태를 유지한다. (아무리 긴 시계열 데이터라도 순전파를 끊지 않고 전파)
        # 긴 시계열 데이터를 처리할 때는 RNN의 은닉상태를 유지해야 한다.
        self.stateful = stateful

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        Wx, Wh, b = self.params
        # N : 미니배치 크기
        # T : T개 분량의 시계열 데이터
        # D : 입력 벡터의 차원 수
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []

        # 출력 값을 담을 그릇
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            # forward() 메서드가 불리면 인스턴스 변수 h에는 마지막 RNN 계층의 은닉 상태가 저장
            # 그래서, 다음번 forward() 메서드 호출 시 stateful이 True이면 먼저 저장된 h값이 그대로 이용.
            # 아니라면, 영행렬로 시작
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            # RNN 계층을 생성하여 집어넣는다.
            layer = RNN(*self.params)
            # hidden layer를 넣고, 받고 반복
            self.h = layer.forward(xs[:, t, :], self.h)
            # return 업데이트
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        # 하류로 흘러보낼 그릇
        dxs = np.empty((N, T, D), dtype='f')
        dh = 0

        # TODO 이거 ... 머리가 아프다.
        grads = [0, 0, 0]
        # 거꾸로 Loop
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh) # 합산된 기울기
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs