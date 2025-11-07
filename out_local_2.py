import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend for plotting

# Set random seed for reproducibility
np.random.seed(520)

# System parameters
A = 0.1
B1 = 2.0
B2 = 1.0
C = 0
D1 = 0
D2 = 0

# Performance index parameters
Q1 = 0.2
Q2 = 0.1
R1, R2 = 1.0, 1.0
theta1, theta2 = 0.6, 0.4
rho1, rho2 = 1.5, 1.6
lambda1, lambda2 = 0.01, 0.01

M = 1000  # Number of data points
# Initialize parameters
W1 = np.array([1, 1, 1])  # Weights for [x^2, x, 1]
W2 = np.array([1, 1, 1])  # Weights for [x^2, x, 1]
alpha = np.zeros(M)
gamma = np.zeros(M)
Sigma_pi = lambda1 / (2 * R1)
Sigma_gamma = lambda2 / (2 * R2)
epsilon = 1e-6


# Generate frequency parameters for probing noise
w1_freqs = np.random.uniform(-100, 100, size=100)
w2_freqs = np.random.uniform(-100, 100, size=100)

# Generate initial data points
dt = 0.05
X = []
x = 0.1  # Initial state (scalar)

for i in range(M):
    t = i * dt
    # Calculate probing noise
    e1 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w1_freqs])
    e2 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w2_freqs])

    # Generate control inputs
    u1 = np.random.normal(alpha[i] + e1, np.sqrt(Sigma_pi))
    u2 = np.random.normal(gamma[i] + e2, np.sqrt(Sigma_gamma))

    # Calculate state increment
    dx = (A * x + B1 * u1 + B2 * u2) * dt  # No noise terms since C, D1, D2 = 0
    x += dx
    X.append(x)

X = np.array(X)

# Helper functions
def phi(x):
    return np.array([x ** 2, x, 1])

def compute_grad_phi(x):
    return np.array([2 * x, 1, 0])

def compute_hessian_phi():
    return np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]])

# Lists to record the values
v1_values = []
v2_values = []
# 添加新的列表来记录每次迭代的点
v1_points = []  # 记录 (X[j], V1_final) 对
v2_points = []  # 记录 (X[j], V2_final) 对

# Main loop
converged = False
s = 0
max_iterations = 100
LAM2_s = np.zeros(M)
alpha_new = np.zeros(M)
gamma_new = np.zeros(M)
while not converged and s < max_iterations:
    # ================== Leader Update ==================
    # Step 2a: Compute LAM1 and LAM2
    d2V2_dx2 = W2[0] * compute_hessian_phi()[0, 0]  # Only x^2 term contributes
    matrix_term = 2 * R2  # D2 = 0

    # Compute LAM1_s
    LAM1_s = -2 * theta2 * R2 / matrix_term  # D1, D2 = 0

    # Compute LAM2_s (using the current state x_m)
    for idx, x_m in enumerate(X):
        dV2_dx_m = W2 @ compute_grad_phi(x_m)
        LAM2_s[idx] = -B2 * dV2_dx_m / matrix_term  # C, D2 = 0

    # Step 2b: Value function update
    xi1 = np.array([phi(x) for x in X]).T
    H1_list = []

    for idx, x_val in enumerate(X):
        alpha_this = alpha[idx]
        gamma_this = gamma[idx]
        term1 = Q1 * x_val ** 2
        term2 = R1 * (alpha_this + theta1 * gamma_this) ** 2 + R1 * Sigma_pi + (theta1 ** 2) * R1 * Sigma_gamma
        term3 = -lambda1 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_pi))
        dV1_dx = W1 @ compute_grad_phi(x_val)
        term4 = dV1_dx * (A * x_val + B1 * alpha_this + B2 * gamma_this)
        H1 = term1 + term2 + term3 + term4
        H1_list.append(H1)

    chi1 = np.array(H1_list) / rho1
    W1_new = np.linalg.pinv(xi1 @ xi1.T) @ xi1 @ chi1  # Use pseudoinverse for stability

    # Step 2c: Policy update
    d2V1_new_dx2 = W1_new[0] * compute_hessian_phi()[0, 0]
    Sigma_pi_new = lambda1 / (2 * R1)  # D1 = 0

    
    for idx, x_m in enumerate(X):
        dV1_dx_m = W1_new @ compute_grad_phi(x_m)
        numerator = (B1 * dV1_dx_m +
                     LAM1_s * B2 * dV1_dx_m +
                     (2 * theta1 * R1 + 2 * theta1 ** 2 * R1 * LAM1_s) * LAM2_s[idx])
        denominator1 = 2 * R1 + 2 * theta1 * R1 * LAM1_s + 2 * theta1 ** 2 * R1 * LAM1_s ** 2
        alpha_new[idx] = -numerator / denominator1

    # ================== Follower Update ==================
    # Step 3a: Value function update
    xi2 = np.array([phi(x) for x in X]).T
    H2_list = []

    for idx, x_val in enumerate(X):
        alpha_this = alpha_new[idx]
        gamma_this = gamma[idx]
        term1 = Q2 * x_val ** 2
        term2 = R2 * (gamma_this + theta2 * alpha_this) ** 2 + R2 * Sigma_gamma + (theta2 ** 2) * R2 * Sigma_pi
        term3 = -lambda2 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_gamma))
        dV2_dx = W2 @ compute_grad_phi(x_val)
        term4 = dV2_dx * (A * x_val + B1 * alpha_this + B2 * gamma_this)
        H2 = term1 + term2 + term3 + term4
        H2_list.append(H2)

    chi2 = np.array(H2_list) / rho2
    W2_new = np.linalg.pinv(xi2 @ xi2.T) @ xi2 @ chi2  # Use pseudoinverse for stability

    # Step 3b: Policy update
    d2V2_new_dx2 = W2_new[0] * compute_hessian_phi()[0, 0]
    Sigma_gamma_new = lambda2 / (2 * R2)  # D2 = 0

    denominator4 = -2 * R2
    denominator5 = 2 * theta2 * R2

    for idx, x_m in enumerate(X):
        dV2_dx_m = W2_new @ compute_grad_phi(x_m)
        denominator6 = B2 * dV2_dx_m
        gamma_new[idx] = (denominator5 * alpha_new[idx] + denominator6) / denominator4

    # Check convergence
    V1_diff = np.linalg.norm(W1_new - W1)
    V2_diff = np.linalg.norm(W2_new - W2)

    if V1_diff < epsilon and V2_diff < epsilon:
        converged = True
    else:
        W1 = W1_new
        W2 = W2_new
        alpha = alpha_new
        gamma = gamma_new
        Sigma_pi = Sigma_pi_new
        Sigma_gamma = Sigma_gamma_new
        s += 1


# 输出结果
print("收敛于迭代次数:", s)
print("最优策略:")
print("领导者策略: N(alpha vector, {:.4f})".format(Sigma_pi))
print("alpha:", alpha)
print("追随者策略: N(gamma vector, {:.4f})".format(Sigma_gamma))
print("gamma:", gamma)
print("值函数参数:")
print("W1:", W1)
print("W2:", W2)
# After convergence, compute final value functions for all states
v1_points = [(x, W1 @ phi(x)) for x in X]
v2_points = [(x, W2 @ phi(x)) for x in X]

# 提取X和Y坐标
v1_x = [point[0] for point in v1_points]
v1_y = [point[1] for point in v1_points]
v2_x = [point[0] for point in v2_points]
v2_y = [point[1] for point in v2_points]

# 对比值函数
def V1_cl(x):
    return 0.1143037 * x**2

def V2_cl(x):
    return 0.0479209 * x**2

# 生成x范围用于绘制对比曲线
x_range = np.linspace(min(min(v1_x), min(v2_x)), max(max(v1_x), max(v2_x)), 100)
v1_cl_y = [V1_cl(x) for x in x_range]
v2_cl_y = [V2_cl(x) for x in x_range]

# 绘制图像，包含散点图和对比曲线
plt.figure(figsize=(8, 5))

# 绘制散点
plt.scatter(v1_x, v1_y, label='Leader Iteration $V_1(x)$', 
           color='blue', alpha=0.6, s=20)
plt.scatter(v2_x, v2_y, label='Follower Iteration $V_2(x)$', 
           color='red', alpha=0.6, s=20)

# 绘制对比曲线
plt.plot(x_range, v1_cl_y, label='Leader $V_1^{cl}(x) = 0.1143037x^2$', 
         color='darkblue', linewidth=3, linestyle='--')
plt.plot(x_range, v2_cl_y, label='Follower $V_2^{cl}(x) = 0.0479209x^2$', 
         color='darkred', linewidth=3, linestyle='--')

plt.xlabel('State x', fontsize=14)
plt.ylabel('Value Function', fontsize=14)
plt.legend(fontsize=17, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 计算误差
v1_errors = [abs(v1_y[i] - V1_cl(v1_x[i]))/V1_cl(v1_x[i]) for i in range(len(v1_x))]
v2_errors = [abs(v2_y[i] - V2_cl(v2_x[i]))/V2_cl(v1_x[i]) for i in range(len(v2_x))]

# 绘制误差图
plt.figure(figsize=(8, 5))
plt.scatter(v1_x, v1_errors, label='Leader Error_1', 
           color='blue', alpha=0.6, s=20)
plt.scatter(v2_x, v2_errors, label='Follower Error_2', 
           color='red', alpha=0.6, s=20)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # 添加y=0参考线
plt.xlabel('State x')
plt.ylabel('Error')
plt.title('Value Function Errors')
plt.legend()
plt.grid(True)
plt.show()

# 计算并显示误差统计信息
print("\nError Statistics:")
print(f"Mean Error_1: {np.mean(v1_errors):.6f}, Std: {np.std(v1_errors):.6f}")
print(f"Mean Error_2: {np.mean(v2_errors):.6f}, Std: {np.std(v2_errors):.6f}")