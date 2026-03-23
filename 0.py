import numpy as np

# 构造 A, B
A = np.zeros((2,3,4), dtype=int)
B = np.zeros((2,3,4), dtype=int)
for a in range(2):
    for m in range(3):
        for n in range(4):
            A[a,m,n] = a*100 + m*10 + n
            B[a,m,n] = a*200 + m*20 + n*2   # 注意这里把 a 当成 b 用，只是为了演示


# 爱因斯坦求和
C = np.einsum('amn,bmn->ab', A, B)
print(C)