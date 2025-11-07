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

# 定义不同的lambda组合
lambda_combinations = [
    (1, 2),  # 较低的信息代价
    (3, 4),  # 原始设置
    (5, 6),  # 较高的信息代价
]

# 存储每种lambda组合的结果
results = {}

# 对每种lambda组合运行算法
for lambda1, lambda2 in lambda_combinations:
    print(f"正在运行 lambda1={lambda1}, lambda2={lambda2} 的情况...")
    
    # 初始化参数
    W1 = np.array([1, 1, 1, 1])
    W2 = np.array([1, 1, 1, 1])
    M = 50  # 数据点数量
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

    # 主循环
    converged = False
    s = 0
    max_iterations = 100
    LAM2_s = np.zeros(M)
    alpha_new = np.zeros(M)
    gamma_new = np.zeros(M)
    
    # 添加价值函数历史记录
    V1_history = []
    V2_history = []
    
    while not converged and s < max_iterations:
        # ================== 领导者更新 ==================
        # Step 2a: 计算LAM1和LAM2
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

        # 记录当前迭代的价值函数值（使用初始状态）
        x0 = np.array([3.0, 3.0])  # 初始状态
        V1_current = W1_new @ phi(x0)  # 领导者价值函数
        V2_current = W2_new @ phi(x0)  # 追随者价值函数
        V1_history.append(V1_current)
        V2_history.append(V2_current)

        # 检查收敛
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

    # 存储结果
    results[(lambda1, lambda2)] = {
        'alpha': alpha[5],  # 使用第6个数据点
        'gamma': gamma[5],  # 使用第6个数据点
        'Sigma_pi': Sigma_pi,
        'Sigma_gamma': Sigma_gamma,
        'iterations': s,
        'V1_history': V1_history,  # 存储领导者价值函数历史
        'V2_history': V2_history   # 存储追随者价值函数历史
    }
    
    print(f"lambda1={lambda1}, lambda2={lambda2} 完成，迭代次数: {s}")
    print(f"领导者策略: N({alpha[5]:.4f}, {Sigma_pi:.4f})")
    print(f"追随者策略: N({gamma[5]:.4f}, {Sigma_gamma:.4f})")

# ========================= 绘制不同lambda组合的最终策略正态分布 ================================
# 生成x轴数据
x = np.linspace(-2, 2, 1000)

# 创建图表
plt.figure(figsize=(8, 5))

# 定义颜色和线型
colors_leader = ['#ff0000', '#ff6666', '#ff9999', '#ffcccc']
colors_follower = ['#000000', '#666666', '#999999', '#cccccc']
line_styles = ['-', '--', '-.', ':']

# 绘制每种lambda组合的正态分布
for idx, (lambda_pair, result) in enumerate(results.items()):
    lambda1, lambda2 = lambda_pair
    alpha_val = result['alpha']
    gamma_val = result['gamma']
    Sigma_pi_val = result['Sigma_pi']
    Sigma_gamma_val = result['Sigma_gamma']
    
    # 计算概率密度函数
    pdf_leader = stats.norm.pdf(x, alpha_val, np.sqrt(Sigma_pi_val))
    pdf_follower = stats.norm.pdf(x, gamma_val, np.sqrt(Sigma_gamma_val))
    
    # 绘制领导者策略
    plt.plot(x, pdf_leader, 
             label=f'Leader ($\lambda_1$={lambda1}, $\lambda_2$={lambda2})', 
             color=colors_leader[idx], 
             linestyle=line_styles[idx],
             linewidth=2)
    
    # 绘制追随者策略
    plt.plot(x, pdf_follower, 
             label=f'Follower ($\lambda_1$={lambda1}, $\lambda_2$={lambda2})', 
             color=colors_follower[idx], 
             linestyle=line_styles[idx],
             linewidth=2)

plt.xlabel('Control Input', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.legend(fontsize=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("所有lambda组合运行完成！")

# ========================= 绘制不同lambda组合的价值函数变化图像（作为子图放在一张大图中） ================================
# 定义颜色
colors = [  '#d11149','#f17105', '#6610f2',"#e6c229"]

# 创建包含两个子图的大图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 第一个子图：领导者价值函数V1
for idx, (lambda_pair, result) in enumerate(results.items()):
    lambda1, lambda2 = lambda_pair
    V1_history = result['V1_history']
    iterations = range(len(V1_history))
    
    # 绘制V1
    ax1.plot(iterations, V1_history, color=colors[idx], marker='*', linewidth=3, 
             label=f'$\lambda_1$={lambda1}, $\lambda_2$={lambda2}')

ax1.set_ylabel('Leader Value Function $V_1$', fontsize=15)
ax1.legend(fontsize=19)
ax1.grid(True, alpha=0.3)

# 第二个子图：追随者价值函数V2
for idx, (lambda_pair, result) in enumerate(results.items()):
    lambda1, lambda2 = lambda_pair
    V2_history = result['V2_history']
    iterations = range(len(V2_history))
    
    # 绘制V2
    ax2.plot(iterations, V2_history, color=colors[idx], marker='o', linewidth=3, 
             label=f'$\lambda_1$={lambda1}, $\lambda_2$={lambda2}')

ax2.set_xlabel('Iteration', fontsize=15)
ax2.set_ylabel('Follower Value Function $V_2$', fontsize=15)
ax2.legend(fontsize=19)
ax2.grid(True, alpha=0.3)

# 调整子图之间的间距
plt.tight_layout()
plt.show()