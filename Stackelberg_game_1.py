import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 使用 Agg 后端进行图像保存
# 设置随机种子以保证结果可重复
np.random.seed(520)

# 系统参数
A = np.array([[0, 1], [0, -1]])
B1 = np.array([[0.3], [0.8]])
B2 = np.array([[0.8], [0.2]])
C = np.array([[1, 0], [0, -1]])
D1 = np.array([[0.3], [0.8]])
D2 = np.array([[-0.8], [0.2]])

# 性能指标参数
Q1 = np.array([[1, 0], [0, 1]])
Q2 = np.array([[5, 0], [0, 5]])
R1, R2 = 5, 8
theta1, theta2 = 0.4, 0.1
rho1, rho2 = 3, 3
lambda1, lambda2 = 0.1, 0.1

M =50  # 数据点数量
# 初始化参数
W1 = np.array([1, 1, 1, 1])
W2 = np.array([1, 1, 1, 1])
alpha = np.zeros(M)
gamma = np.zeros(M)
Sigma_pi = lambda1 / (2 * R1)
Sigma_gamma = lambda2 / (2 * R2)
epsilon = 1e-3

# 生成探测噪声的频率参数
w1_freqs = np.random.uniform(-100, 100, size=100)
w2_freqs = np.random.uniform(-100, 100, size=100)

# 生成初始数据点
dt = 0.01
X = []
x = np.array([3.0, 3.0])  # 初始状态

for i in range(M):
    t = i * dt
    # 计算探测噪声
    e1 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w1_freqs])
    e2 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w2_freqs])

    # 生成控制输入
    u1 = np.random.normal(alpha[i] + e1, np.sqrt(Sigma_pi))
    u2 = np.random.normal(gamma[i] + e2, np.sqrt(Sigma_gamma))

    # 计算状态增量
    deterministic = (A @ x.reshape(-1, 1) + B1 * u1 + B2 * u2) * dt
    noise_C = C @ x.reshape(-1, 1) * np.sqrt(dt) * np.random.normal()
    noise_D1 = D1 * u1 * np.sqrt(dt) * np.random.normal()
    noise_D2 = D2 * u2 * np.sqrt(dt) * np.random.normal()

    dx = deterministic.flatten() + noise_C.flatten() + noise_D1.flatten() + noise_D2.flatten()
    x += dx
    X.append(x.copy())

X = np.array(X)
print("系统数据点为:", X)

# 辅助函数定义
def phi(x):
    return np.array([x[0] ** 2, x[0] * x[1], x[1] ** 2, 1])


def compute_grad_phi(x):
    return np.array([
        [2 * x[0], 0],
        [x[1], x[0]],
        [0, 2 * x[1]],
        [0, 0]
    ])


def compute_hessian_phi():
    return [
        np.array([[2, 0], [0, 0]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, 0], [0, 2]]),
        np.array([[0, 0], [0, 0]])
    ]


hessians = compute_hessian_phi()


def compute_dVdx(W, x):
    grad_phi = compute_grad_phi(x)
    return W.T @ grad_phi


def compute_d2Vdx2(W):
    return sum(W[i] * hessians[i] for i in range(4))

W1_history = []
W2_history = []
alpha_history = []
gamma_history = []
Sigma_pi_history = []
Sigma_gamma_history = []

# 主循环
converged = False
s = 0
max_iterations = 100
LAM2_s = np.zeros(M)
alpha_new = np.zeros(M)
gamma_new = np.zeros(M)
while not converged and s < max_iterations:
    # ================== 领导者更新 ==================
    # Step 2a: 计算LAM1和LAM2
    # 使用追随者当前的值函数参数W2
    d2V2_dx2 = compute_d2Vdx2(W2)
    matrix_term = (2 * R2 + D2.T @ d2V2_dx2 @ D2).item()

    # 计算LAM1_s
    LAM1_s = -(2 * theta2 * R2 + D2.T @ d2V2_dx2 @ D1).item() / matrix_term

    # 计算LAM2_s（为每个状态计算）
    
    for j, x_val in enumerate(X):
        dV2_dx = compute_dVdx(W2, x_val).reshape(-1, 1)
        term = (D2.T @ d2V2_dx2 @ C @ x_val.reshape(-1, 1) + B2.T @ dV2_dx).item()
        LAM2_s[j] = -term / matrix_term

    # Step 2b: 值函数更新
    xi1 = np.array([phi(x) for x in X]).T
    H1_list = []

    for idx, x_val in enumerate(X):
        x = x_val.reshape(-1, 1)
        alpha_this = alpha[idx]
        gamma_this = gamma[idx]
        term1 = (x.T @ Q1 @ x).item()
        term2 = R1 * (alpha_this + theta1 * gamma_this) ** 2 + R1 * Sigma_pi + (theta1 ** 2) * R1 * Sigma_gamma
        term3 = -lambda1 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_pi))

        dV1_dx = compute_dVdx(W1, x_val.flatten()).reshape(-1, 1)
        term4 = (dV1_dx.T @ (A @ x + B1 * alpha_this + B2 * gamma_this)).item()

        d2V1_dx2 = compute_d2Vdx2(W1)
        term5 = 0.5 * (x.T @ C.T @ d2V1_dx2 @ C @ x).item()
        term6 = 0.5 * (D1.T @ d2V1_dx2 @ D1).item() * (alpha_this ** 2 + Sigma_pi)
        term7 = 0.5 * (D2.T @ d2V1_dx2 @ D2).item() * (gamma_this ** 2 + Sigma_gamma)
        term8 = (x.T @ C.T @ d2V1_dx2 @ D1).item() * alpha_this
        term9 = (x.T @ C.T @ d2V1_dx2 @ D2).item() * gamma_this
        term10 = alpha_this * (D1.T @ d2V1_dx2 @ D2).item() * gamma_this

        H1 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
        H1_list.append(H1)

    chi1 = np.array(H1_list) / rho1
    
    W1_new = np.linalg.inv(xi1 @ xi1.T) @ xi1 @ chi1  #如果不可逆时候用伪逆np.linalg.pinv(xi1 @ xi1.T)

    # Step 2c: 策略更新
    d2V1_new_dx2 = compute_d2Vdx2(W1_new)
    Sigma_pi_new = lambda1 / (2 * R1 + (D1.T @ d2V1_new_dx2 @ D1).item())

    # 计算新alpha（为每个状态计算）
    for j, x_val in enumerate(X):
        x = x_val.reshape(-1, 1)
        dV1_dx = compute_dVdx(W1_new, x_val).reshape(-1, 1)

        denominator1 = (2 * R1 + (D1.T @ d2V1_new_dx2 @ D1).item() + (D1.T @ d2V1_new_dx2 @ D2).item() * LAM1_s + LAM1_s * (D2.T @ d2V1_new_dx2 @ D2).item() * LAM1_s + 2 * theta1 ** 2 * LAM1_s * R1 * LAM1_s + 4 * theta1 * R1 * LAM1_s + (D2.T @ d2V1_new_dx2 @ D1).item() * LAM1_s)

        numerator = ((B1.T @ dV1_dx).item() +
                     LAM1_s * (B2.T @ dV1_dx).item() +
                     (D1.T @ d2V1_new_dx2 @ C @ x).item() +
                     LAM1_s * (D2.T @ d2V1_new_dx2 @ C @ x).item() +
                     ((D2.T @ d2V1_new_dx2 @ D2).item() * LAM1_s + 2 * theta1 * R1 +
                      2 * theta1 ** 2 * R1 * LAM1_s + (D1.T @ d2V1_new_dx2 @ D2).item()) * LAM2_s[j])

        alpha_new[j] = -numerator / denominator1

    # ================== 追随者更新 ==================
    # Step 3a: 值函数更新
    xi2 = np.array([phi(x) for x in X]).T
    H2_list = []

    for idx, x_val in enumerate(X):
        x = x_val.reshape(-1, 1)
        alpha_this = alpha_new[idx]
        gamma_this = gamma[idx]
        term1 = (x.T @ Q2 @ x).item()
        term2 = R2 * (gamma_this + theta2 * alpha_this) ** 2 + R2 * Sigma_gamma + (theta2 ** 2) * R2 * Sigma_pi
        term3 = -lambda2 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_gamma))

        dV2_dx = compute_dVdx(W2, x_val.flatten()).reshape(-1, 1)
        term4 = (dV2_dx.T @ (A @ x + B1 * alpha_this + B2 * gamma_this)).item()

        d2V2_dx2 = compute_d2Vdx2(W2)
        term5 = 0.5 * (x.T @ C.T @ d2V2_dx2 @ C @ x).item()
        term6 = 0.5 * (D1.T @ d2V2_dx2 @ D1).item() * (alpha_this ** 2 + Sigma_pi)
        term7 = 0.5 * (D2.T @ d2V2_dx2 @ D2).item() * (gamma_this ** 2 + Sigma_gamma)
        term8 = (x.T @ C.T @ d2V2_dx2 @ D1).item() * alpha_this
        term9 = (x.T @ C.T @ d2V2_dx2 @ D2).item() * gamma_this
        term10 = alpha_this * (D1.T @ d2V2_dx2 @ D2).item() * gamma_this

        H2 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
        H2_list.append(H2)

    chi2 = np.array(H2_list) / rho2
    W2_new = np.linalg.inv(xi2 @ xi2.T) @ xi2 @ chi2  #如果不可逆时候用伪逆np.linalg.pinv(xi2 @ xi2.T)

    # Step 3b: 策略更新
    d2V2_new_dx2 = compute_d2Vdx2(W2_new)
    Sigma_gamma_new = lambda2 / (2 * R2 + (D2.T @ d2V2_new_dx2 @ D2).item())

    denominator4 = -(2 * R2 + (D2.T @ d2V2_new_dx2 @ D2).item())
    denominator5 = (2 * theta2 * R2 + (D2.T @ d2V2_new_dx2 @ D1).item())

    # 计算新gamma（为每个状态计算）
    for j, x_val in enumerate(X):
        x_m = x_val.reshape(-1, 1)
        dV2_dx_m = compute_dVdx(W2_new, x_val).reshape(-1, 1)
        denominator6 = ((D2.T @ d2V2_new_dx2 @ C @ x_m).item() + (B2.T @ dV2_dx_m).item())
        gamma_new[j] = denominator5 * alpha_new[j] / denominator4 + denominator6 / denominator4

    # 检查收敛
    V1_diff = np.linalg.norm(W1_new - W1)
    V2_diff = np.linalg.norm(W2_new - W2)
    print("V1_diff为:", V1_diff)
    print("V2_diff为:", V2_diff)

    # 保存历史记录
    W1_history.append(W1.copy())
    W2_history.append(W2.copy())
    alpha_history.append(alpha.copy())
    gamma_history.append(gamma.copy())
    Sigma_pi_history.append(Sigma_pi)
    Sigma_gamma_history.append(Sigma_gamma)

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

# ========================= 计算KL散度 ==============================
def kl_divergence_normal(mu1, sigma1, mu2, sigma2):
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5
# 使用最后一次迭代的结果作为目标分布
mu_leader_target = alpha_history[-1]
sigma_leader_target = np.sqrt(Sigma_pi_history[-1])
mu_follower_target = gamma_history[-1]
sigma_follower_target = np.sqrt(Sigma_gamma_history[-1])

# 计算每次迭代的KL散度
kl_leader = []
kl_follower = []

for i in range(len(alpha_history)):
    # 领导者KL散度
    mu_leader = alpha_history[i]
    sigma_leader = np.sqrt(Sigma_pi_history[i])
    kl_leader.append(kl_divergence_normal(mu_leader, sigma_leader, mu_leader_target, sigma_leader_target))
    
    # 跟随者KL散度
    mu_follower = gamma_history[i]
    sigma_follower = np.sqrt(Sigma_gamma_history[i])
    kl_follower.append(kl_divergence_normal(mu_follower, sigma_follower, mu_follower_target, sigma_follower_target))

# ========================= 绘制KL散度变化图 ============================
plt.figure(figsize=(8, 4.5))
iterations = range(len(kl_leader))
plt.plot(iterations, [kl[5] for kl in kl_leader], color="#ff0000", marker='*', linewidth=3, label='Leader KL Divergence')
plt.plot(iterations, [kl[5] for kl in kl_follower], color="#000000", marker='*', linewidth=3, label='Follower KL Divergence')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss Value', fontsize=12)
plt.legend(loc='best', fontsize=17)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数尺度更好地显示收敛
plt.tight_layout()
plt.show()

# ========================= 绘制权重图 ================================
# 将历史记录转换为数组
W1_history = np.array(W1_history)
W2_history = np.array(W2_history)

# 创建两个子图
plt.figure(figsize=(14, 5))

# 定义颜色和线型
colors = ["#e6c229", '#f17105', '#d11149', '#6610f2']

# 第一张图：W1各分量
plt.subplot(1, 2, 1)
for i in range(4):
    plt.plot(W1_history[:, i], label=f'$W_1$$_{i+1}$', color=colors[i], marker='*', linewidth=3)

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Weight of NNs', fontsize=12)
plt.legend(loc='best', fontsize=17)
plt.grid(True, alpha=0.3)

# 第二张图：W2各分量
plt.subplot(1, 2, 2)
for i in range(4):
    plt.plot(W2_history[:, i], label=f'$W_2$$_{i+1}$', color=colors[i], marker='*', linewidth=3)

plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Weight of NNs', fontsize=12)
plt.legend(loc='best', fontsize=17)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================= 绘制最终策略的正态分布 ================================
# 生成x轴数据
x = np.linspace(-2, 2, 1000)

# 计算概率密度函数
pdf_leader = stats.norm.pdf(x, alpha[5], Sigma_pi)
pdf_follower = stats.norm.pdf(x, gamma[5], Sigma_gamma)

# 创建图表
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_leader, label=f'Leader: N({alpha[5]:.2f}, {Sigma_pi:.2f})', color="#ff0000", linewidth=3, linestyle='-')
plt.plot(x, pdf_follower, label=f'Follower: N({gamma[5]:.2f}, {Sigma_gamma:.2f})', color="#000000", linewidth=3, linestyle='--')

plt.xlabel('Control Input', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(title=f'$\lambda_1$={lambda1},$\lambda_2$={lambda2}', fontsize=16,title_fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========================= 绘制价值函数变化图像 ================================
# 假设使用第6个数据点，计算每次迭代的V1和V2
V1_values = [W.dot(phi(X[6])) for W in W1_history]
V2_values = [W.dot(phi(X[6])) for W in W2_history]

# 创建迭代次数列表
iterations = range(len(V1_values))
# 绘制价值函数变化曲线
plt.figure(figsize=(8, 4.2))
plt.plot(iterations, V1_values, color="#ff0000", marker='*', linewidth=3, label='Leader $V_1$')
plt.plot(iterations, V2_values, color="#000000", marker='*', linewidth=3, label='Follower $V_2$')
plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Value Function', fontsize=14)
plt.legend(title=f'$\lambda_1$={lambda1},$\lambda_2$={lambda2},$\Theta_1$={theta1},$\Theta_2$={theta2}', fontsize=16,title_fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# ================================= hcd====================================