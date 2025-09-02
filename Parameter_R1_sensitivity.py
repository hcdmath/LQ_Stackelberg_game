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
Q1_values = [
    np.array([[0.1, 0], [0, 0.1]]),
    np.array([[0.2, 0], [0, 0.2]]),
    np.array([[0.3, 0], [0, 0.3]]),
    np.array([[0.5, 0], [0, 0.5]])
]  # 不同的Q1值
Q2 = np.array([[0.5, 0], [0, 0.5]])
R1, R2 = 0.5, 0.5
theta1, theta2 = 0.1, 0.1
rho1, rho2 = 3, 3
lambda1, lambda2 = 10, 10

# 初始化参数
M = 50  # 数据点数量
dt = 0.01
epsilon = 1e-3
max_iterations = 100

# 存储不同Q1的结果
results = {}

for Q1 in Q1_values:
    print(f"正在计算 Q1 = {Q1} 的情况...")
    
    # 初始化参数
    W1 = np.array([1, 1, 1, 1])
    W2 = np.array([1, 1, 1, 1])
    alpha = 0
    gamma = 0
    Sigma_pi = lambda1 / (2 * R1)
    Sigma_gamma = lambda2 / (2 * R2)
    
    # 生成探测噪声的频率参数
    w1_freqs = np.random.uniform(-100, 100, size=100)
    w2_freqs = np.random.uniform(-100, 100, size=100)
    
    # 生成初始数据点
    X = []
    x = np.array([3.0, 3.0])  # 初始状态
    
    for _ in range(M):
        t = _ * dt
        # 计算探测噪声
        e1 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w1_freqs])
        e2 = 0.01 * np.sum([np.sin(w * t) * np.exp(-0.05 * t) for w in w2_freqs])
    
        # 生成控制输入
        u1 = np.random.normal(alpha + e1, np.sqrt(Sigma_pi))
        u2 = np.random.normal(gamma + e2, np.sqrt(Sigma_gamma))
    
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
    
    W1_history = []
    W2_history = []
    alpha_history = []
    gamma_history = []
    Sigma_pi_history = []
    Sigma_gamma_history = []
    
    while not converged and s < max_iterations:
        # ================== 领导者更新 ==================
        # Step 2a: 计算LAM1和LAM2
        # 使用追随者当前的值函数参数W2
        d2V2_dx2 = compute_d2Vdx2(W2)
        matrix_term = 2 * R2 + D2.T @ d2V2_dx2 @ D2
    
        # 计算LAM1_s
        LAM1_s = -(2 * theta2 * R2 + D2.T @ d2V2_dx2 @ D1)/matrix_term
    
        # 计算LAM2_s（使用第6个状态x[6]）
        x_m = X[6].T
        dV2_dx_m = compute_dVdx(W2, x_m)
        LAM2_s = -(D2.T @ d2V2_dx2 @ C @ x_m + B2.T @ dV2_dx_m)/matrix_term
    
        # Step 2b: 值函数更新
        xi1 = np.array([phi(x) for x in X]).T
        H1_list = []
    
        for x in X:
            # 计算各项
            x = x.reshape(-1, 1)
            term1 = x.T @ Q1 @ x  # 这里使用当前的Q1值
            term2 = R1 * (alpha + theta1 * gamma) ** 2 + R1 * Sigma_pi + (theta1 ** 2) * R1 * Sigma_gamma
            term3 = -lambda1 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_pi))
    
            dV1_dx = compute_dVdx(W1, x.flatten())
            term4 = dV1_dx @ (A @ x + B1 * alpha + B2 * gamma)
    
            d2V1_dx2 = compute_d2Vdx2(W1)
            term5 = 0.5 * x.T @ C.T @ d2V1_dx2 @ C @ x
            term6 = 0.5 * D1.T @ d2V1_dx2 @ D1 * (alpha ** 2 + Sigma_pi)
            term7 = 0.5 * D2.T @ d2V1_dx2 @ D2 * (gamma ** 2 + Sigma_gamma)
            term8 = x.T @ C.T @ d2V1_dx2 @ D1 * alpha
            term9 = x.T @ C.T @ d2V1_dx2 @ D2 * gamma
            term10 = alpha * D1.T @ d2V1_dx2 @ D2 * gamma
    
            H1 = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10).item()
            H1_list.append(H1)
    
        chi1 = np.array(H1_list) / rho1
        
        W1_new = np.linalg.inv(xi1 @ xi1.T) @ xi1 @ chi1  #如果不可逆时候用伪逆np.linalg.pinv(xi1 @ xi1.T)
    
        # Step 2c: 策略更新
        d2V1_new_dx2 = compute_d2Vdx2(W1_new)
        Sigma_pi_new = lambda1 / (2 * R1 + D1.T @ d2V1_new_dx2 @ D1).item()
    
        # 计算新alpha
        denominator1 = (2 * R1 + D1.T @ d2V1_new_dx2 @ D1+ D1.T @ d2V1_new_dx2 @ D2* LAM1_s+ LAM1_s * D2.T @ d2V1_new_dx2 @ D2 * LAM1_s+ 2 * theta1 ** 2 * LAM1_s * R1 * LAM1_s+ 4 * theta1 * R1* LAM1_s+ D2.T @ d2V1_new_dx2 @ D1* LAM1_s)
        numerator = (B1.T @ compute_dVdx(W1_new, x_m) +
                     LAM1_s * B2.T @ compute_dVdx(W1_new, x_m) +
                     D1.T @ d2V1_new_dx2 @ C @ x_m.reshape(-1, 1) +
                     LAM1_s * D2.T @ d2V1_new_dx2 @ C @ x_m.reshape(-1, 1) +
                     (D2.T @ d2V1_new_dx2 @ D2 * LAM1_s + 2 * theta1 * R1 +
                      2 * theta1 ** 2 * R1 * LAM1_s + D1.T @ d2V1_new_dx2 @ D2) * LAM2_s)
    
        alpha_new = -numerator / denominator1
    
        # ================== 追随者更新 ==================
        # Step 3a: 值函数更新
        xi2 = np.array([phi(x) for x in X]).T
        H2_list = []
    
        for x in X:
            x = x.reshape(-1, 1)
            term1 = x.T @ Q2 @ x
            term2 = R2 * (gamma + theta2 * alpha_new) ** 2 + R2 * Sigma_gamma + (theta2 ** 2) * R2 * Sigma_pi
            term3 = -lambda2 / 2 * (np.log(2 * np.pi * np.e) + np.log(Sigma_gamma))
    
            dV2_dx = compute_dVdx(W2, x.flatten())
            term4 = dV2_dx @ (A @ x + B1 * alpha_new + B2 * gamma)
    
            d2V2_dx2 = compute_d2Vdx2(W2)
            term5 = 0.5 * x.T @ C.T @ d2V2_dx2 @ C @ x
            term6 = 0.5 * D1.T @ d2V2_dx2 @ D1 * (alpha_new ** 2 + Sigma_pi)
            term7 = 0.5 * D2.T @ d2V2_dx2 @ D2 * (gamma ** 2 + Sigma_gamma)
            term8 = x.T @ C.T @ d2V2_dx2 @ D1 * alpha_new
            term9 = x.T @ C.T @ d2V2_dx2 @ D2 * gamma
            term10 = alpha_new * D1.T @ d2V2_dx2 @ D2 * gamma
    
            H2 = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10).item()
            H2_list.append(H2)
    
        chi2 = np.array(H2_list) / rho2
        W2_new = np.linalg.inv(xi2 @ xi2.T) @ xi2 @ chi2  #如果不可逆时候用伪逆np.linalg.pinv(xi2 @ xi2.T)
    
        # Step 3b: 策略更新
        d2V2_new_dx2 = compute_d2Vdx2(W2_new)
        Sigma_gamma_new = lambda2 / (2 * R2 + D2.T @ d2V2_new_dx2 @ D2).item()
    
        denominator4 = -(2 * R2 + D2.T @ d2V2_new_dx2 @ D2)
        denominator5 = (2 * theta2 * R2 + D2.T @ d2V2_new_dx2 @ D1)
        denominator6 = D2.T @ d2V2_new_dx2 @ C @ x_m + B2.T @ compute_dVdx(W2_new, x_m)
        gamma_new = denominator5 * alpha_new/denominator4 + denominator6 / denominator4
    
        # 检查收敛
        V1_diff = np.linalg.norm(W1_new - W1)
        V2_diff = np.linalg.norm(W2_new - W2)
    
        # 保存历史记录
        W1_history.append(W1.copy())
        W2_history.append(W2.copy())
        alpha_history.append(alpha)
        gamma_history.append(gamma)
        Sigma_pi_history.append(Sigma_pi)
        Sigma_gamma_history.append(Sigma_gamma)
    
        if V1_diff < epsilon and V2_diff < epsilon:
            converged = True
        else:
            W1 = W1_new
            W2 = W2_new
            alpha = alpha_new.item()
            gamma = gamma_new.item()
            Sigma_pi = Sigma_pi_new
            Sigma_gamma = Sigma_gamma_new
            s += 1
    
    # 存储结果（包括领导者和跟随者的信息）
    results[str(Q1)] = {
        'alpha': alpha,
        'gamma': gamma,
        'Sigma_pi': Sigma_pi,
        'Sigma_gamma': Sigma_gamma,
        'W1_history': W1_history,
        'W2_history': W2_history,
        'V1_values': [W.dot(phi(X[6])) for W in W1_history],
        'V2_values': [W.dot(phi(X[6])) for W in W2_history]
    }
    
    print(f"Q1 = {Q1} 计算完成，迭代次数: {s}")
    print(f"领导者策略: N({alpha:.4f}, {Sigma_pi:.4f})")
    print(f"跟随者策略: N({gamma:.4f}, {Sigma_gamma:.4f})")

# ========================= 绘制不同Q1的领导者和跟随者的最优策略图 ================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, len(Q1_values)))

# 生成x轴数据
x = np.linspace(-50, 50, 1000)

# 领导者策略图
for i, Q1 in enumerate(Q1_values):
    result = results[str(Q1)]
    pdf_leader = stats.norm.pdf(x, result['alpha'], result['Sigma_pi'])
    ax1.plot(x, pdf_leader, label=f'Q1={Q1[0,0]}I', color=colors[i], linewidth=2.5)

ax1.set_title('Leader Optimal Policies for Different Q1 Values', fontsize=14)
ax1.set_xlabel('Control Input', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.legend(title='Q1 Values', loc='upper right', fontsize=10)
ax1.grid(True)

# 跟随者策略图
for i, Q1 in enumerate(Q1_values):
    result = results[str(Q1)]
    pdf_follower = stats.norm.pdf(x, result['gamma'], result['Sigma_gamma'])
    ax2.plot(x, pdf_follower, label=f'Q1={Q1[0,0]}I', color=colors[i], linewidth=2.5)

ax2.set_title('Follower Optimal Policies for Different Q1 Values', fontsize=14)
ax2.set_xlabel('Control Input', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.legend(title='Q1 Values', loc='upper right', fontsize=10)
ax2.grid(True)

plt.tight_layout()
plt.show()

# ========================= 绘制不同Q1的领导者和跟随者的值函数变化图 ================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 领导者值函数图
for i, Q1 in enumerate(Q1_values):
    result = results[str(Q1)]
    iterations = range(len(result['V1_values']))
    ax1.plot(iterations, result['V1_values'], label=f'Q1={Q1[0,0]}I', color=colors[i], linewidth=3)

ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Value Function V1', fontsize=12)
ax1.set_title('Convergence of Leader Value Function for Different Q1 Values', fontsize=14)
ax1.legend(title='Q1 Values', loc='upper right')
ax1.grid(True)

# 跟随者值函数图
for i, Q1 in enumerate(Q1_values):
    result = results[str(Q1)]
    iterations = range(len(result['V2_values']))
    ax2.plot(iterations, result['V2_values'], label=f'Q1={Q1[0,0]}I', color=colors[i], linewidth=3)

ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Value Function V2', fontsize=12)
ax2.set_title('Convergence of Follower Value Function for Different Q1 Values', fontsize=14)
ax2.legend(title='Q1 Values', loc='upper right')
ax2.grid(True)

plt.tight_layout()
plt.show()

# ========================= 绘制Q1对最终策略参数的影响 ================================
q1_diag_values = [Q1[0, 0] for Q1 in Q1_values]
alpha_list = [results[str(Q1)]['alpha'] for Q1 in Q1_values]
gamma_list = [results[str(Q1)]['gamma'] for Q1 in Q1_values]
sigma_pi_list = [results[str(Q1)]['Sigma_pi'] for Q1 in Q1_values]
sigma_gamma_list = [results[str(Q1)]['Sigma_gamma'] for Q1 in Q1_values]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 均值参数图
ax1.plot(q1_diag_values, alpha_list, 'o-', linewidth=3, markersize=8, label='Leader Mean (α)')
ax1.plot(q1_diag_values, gamma_list, 's-', linewidth=3, markersize=8, label='Follower Mean (γ)')
ax1.set_xlabel('Q1 Diagonal Value', fontsize=12)
ax1.set_ylabel('Policy Mean', fontsize=12)
ax1.set_title('Policy Means vs Q1', fontsize=14)
ax1.legend()
ax1.grid(True)

# 方差参数图
ax2.plot(q1_diag_values, sigma_pi_list, 'o-', linewidth=3, markersize=8, label='Leader Variance (Σπ)')
ax2.plot(q1_diag_values, sigma_gamma_list, 's-', linewidth=3, markersize=8, label='Follower Variance (Σγ)')
ax2.set_xlabel('Q1 Diagonal Value', fontsize=12)
ax2.set_ylabel('Policy Variance', fontsize=12)
ax2.set_title('Policy Variances vs Q1', fontsize=14)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
