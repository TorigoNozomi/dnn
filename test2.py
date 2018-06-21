import parameter_initialization as PI
import numpy as np



useinit = 'randn'

w = PI.he(10,10)
w_1 = np.random.randn(10, 10) * np.sqrt(2.0 / 10)

print(w)
print(w_1)