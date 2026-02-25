#円形流路1.5 mmの電場解析
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import matplotlib.colors as mcolors


# --- パラメータ設定 ---
R = 0.75  # 半径 [mm]
Nr = 76  # 半径方向分割数
Ntheta = 101  # 角度方向分割数
r = np.linspace(0, R, Nr)
theta = np.linspace(np.pi/2, 5/2*np.pi, Ntheta, endpoint=True)
dr = r[1] - r[0]
dtheta = theta[1] - theta[0]
R_grid, Theta_grid = np.meshgrid(r, theta, indexing='ij')

# デカルト変換
X = R_grid * np.cos(Theta_grid)
Y = R_grid * np.sin(Theta_grid)

# --- ファイルの保存名 ---
save_path = "laplace_polar_result1.5 mm.npz"

# --- 電極定義 ---
electrode_angle_width = np.pi / 36  # 約5度
electrode_r_min = 0.1
electrode_r_max = 0.75

#保存しているファイルの削除
#if os.path.exists(save_path):
   #os.remove(save_path)  # ← これを追加して、保存ファイルを消す


# --- 計算または読み込み ---
if os.path.exists(save_path):
    print("保存済みデータを読み込み中...")
    data = np.load(save_path)
    phi = data["phi"]
    Ex = data["Ex"]
    Ey = data["Ey"]
    E_mag = data["E_mag"]
else:
    print("新たに計算を開始します...")
    # 電位初期化
    phi = np.zeros((Nr, Ntheta))
    # 電極マスク
    electrode_mask = np.zeros_like(phi, dtype=bool)

    # 左（+1V）
    left_mask = (Theta_grid > np.pi - electrode_angle_width/2) & \
                (Theta_grid < np.pi + electrode_angle_width/2) & \
                (R_grid >= electrode_r_min) & (R_grid <= electrode_r_max)
    phi[left_mask] = 95
    electrode_mask |= left_mask

    # 右（-1V）
    right_mask = (Theta_grid > 2*np.pi - electrode_angle_width/2) & \
                 (Theta_grid < 2*np.pi + electrode_angle_width/2) & \
                 (R_grid >= electrode_r_min) & (R_grid <= electrode_r_max)
    phi[right_mask] = -95
    electrode_mask |= right_mask

    # --- ラプラス方程式の反復解法（Neumann境界） ---
    def solve_laplace_neumann_polar(phi, max_iter=10000, tol=1e-6):
        for it in range(max_iter):
            phi_old = phi.copy()
            for i in range(1, Nr-1):
                for j in range(Ntheta):
                    if electrode_mask[i, j]:
                        continue
                    jp = (j+1) % Ntheta
                    jm = (j-1) % Ntheta
                    r_val = r[i]
                    phi[i, j] = (
                                   (1/dr**2 + 1/(2*r_val*dr)) * phi[i+1, j]
                                 + (1/dr**2 - 1/(2*r_val*dr)) * phi[i-1, j]
                                 + (phi[i, jp] + phi[i, jm]) / (r_val**2 * dtheta**2)
                  ) / (2/dr**2 + 2/(r_val**2 * dtheta**2))
            phi[Nr-1, :] = phi[Nr-2, :]  # Neumann境界
            phi[0, :] = phi[1, :]        # 原点 Neumann（追加)
            if np.max(np.abs(phi - phi_old)) < tol:
                print(f"収束しました（{it} 回）")
                break
        return phi

    phi = solve_laplace_neumann_polar(phi)


    # --- 電場計算 ---
    Er = np.zeros_like(phi)
    Etheta = np.zeros_like(phi)
    for i in range(1, Nr-1):
        Er[i, :] = -(phi[i+1,:] - phi[i-1,:]) / (2 * dr)
    for j in range(Ntheta):
        jp = (j+1) % Ntheta
        jm = (j-1) % Ntheta
        Etheta[:, j] = -(phi[:, jp] - phi[:, jm]) / (2 * dtheta * r)

    Ex = Er * np.cos(Theta_grid) - Etheta * np.sin(Theta_grid)
    Ey = Er * np.sin(Theta_grid) + Etheta * np.cos(Theta_grid)
    E_mag = np.sqrt(Ex**2 + Ey**2)

    # --- 保存 ---
    np.savez_compressed(save_path, phi=phi, Ex=Ex, Ey=Ey, E_mag=E_mag)
    print("計算結果を保存しました。")

    # --- 安全な電場強度マップ描画 ---
# NaN/inf を除去
E_mag = np.nan_to_num(E_mag, nan=0.0, posinf=0.0, neginf=0.0)

# --- 2D 電場強度マップ（等高線図） ---
plt.figure(figsize=(6, 5))
levels = np.arange(0, 1320, 120)# 明示的にレベルを定義
norm = mcolors.Normalize(vmin=0, vmax=10)
contour = plt.contourf(X, Y, E_mag, levels=levels, cmap='inferno')
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
plt.title(r'|$E$| (V/mm)')
plt.colorbar(contour, label='|E| (V/mm)')
plt.axis('equal')

# 描画範囲を -1~1 に固定
plt.xlim(-1.0, 1.0)
plt.ylim(-1.0, 1.0)

plt.tight_layout()
plt.show()

mask = (X >= -0.1) & (X <= 0.1) & (Y >= 0.0) & (Y <= 0.2)
E_avg = np.mean(E_mag[mask])
print(f"平均電場: {E_avg} V/mm")

mask = (X >= -0.1) & (X <= 0.1) & (Y >= 0) & (Y <= 0.15)
E_avg = np.mean(E_mag[mask])
print(f"-0.1<x<0.1 0<y<0.15において: {E_avg:.3f} V/mm")

Emax=np.max(E_mag)
print(f"最大電場: {Emax} V/mm")

E_avg_total = np.mean(E_mag)
print(f"電場全体の平均値: {E_avg_total:.3f} V/mm")

# --- 興味ある領域の指定 ---
x_min, x_max = -0.1, 0.1  # mm
y_min, y_max = -0.25, -0.05   # mm

# --- 領域マスク作成 ---
region_mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

# --- 領域内の電場平均値 ---
mean_E = np.nanmean(E_mag[region_mask])
print(f"領域({x_min}〜{x_max} mm, {y_min}〜{y_max} mm)の電場の平均値: {mean_E:.4f} V/mm")
print(X.shape, Y.shape, E_mag.shape)

left_mask  = X < 0
right_mask = X > 0
E_max_left  = np.max(E_mag[left_mask])
E_max_right = np.max(E_mag[right_mask])

print(f"x<0 側の最大電場: {E_max_left:.6f} V/mm")
print(f"x>0 側の最大電場: {E_max_right:.6f} V/mm")
# --- 興味ある領域の指定 ---
x_min, x_max = -0.1, 0.1  # mm
y_min, y_max = -0.22, -0.02   # mm

# --- 領域マスク作成 ---
region_mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

# --- 領域内の電場平均値 ---
mean_E = np.nanmean(E_mag[region_mask])
print(f"領域({x_min}〜{x_max} mm, {y_min}〜{y_max} mm)の電場の平均値: {mean_E:.4f} V/mm")