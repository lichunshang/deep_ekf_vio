import numpy as np
import se3

A = np.array([[-0.9954716, -0.00195916, -0.09504104],
              [0.0190242, -0.98366803, -0.17898448],
              [-0.09313816, -0.17998198, 0.97925067]])
# A = se3.exp_SO3([0.001, -0.002, 0.003])

log_A = np.zeros([3, 3,])

for i in range(1, 50):
    log_A = log_A + ((-1) ** i / i) * np.linalg.matrix_power(A - np.eye(3, 3), i)

print(log_A)
print(se3.unskew3(log_A))