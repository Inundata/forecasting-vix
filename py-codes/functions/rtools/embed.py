import numpy as np

def embed(Y, lag):
    # for the purpose to replicate R's embed function

    # 아래 np.hstack은 embed와 동일함.
    # 예시) X가 10개의 열을 가지고 있는 경우, embed(X, 2)라고 하면,
    # 첫 10개의 열은 t기의 information을
    # 그 다음 11 ~ 20개의 열은 t-1부터 시작한다.
    aux = np.hstack([Y[i:-(lag-i) if lag-i != 0 else None] for i in range(lag+1)[::-1]])

    return aux