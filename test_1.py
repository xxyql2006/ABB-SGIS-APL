import numpy as np
cur = 621
t_stable = (cur/630)**1.66*42
theta_n_1 = 24.3 - 5
t_amb = 18.4
theta_stable = t_stable + t_amb
# theta_n = theta_n_1 + (theta_stable-theta_n_1)*(1-np.exp(-300/5500))
exp_cont = 0.94692
theta_n = theta_n_1 + (theta_stable-theta_n_1)*(1-exp_cont)
print(theta_n+5)