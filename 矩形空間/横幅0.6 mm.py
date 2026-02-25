import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
import matplotlib.colors as mcolors  # ← これを追加

# --- パラメータ ---
W = 0.6   # 横幅 (mm)
H = 1.5   # 縦幅 (mm)
d = 0.2   # 電極間距離 (mm)
e_length = 0.2  # 電極の長さ (mm)
e_width = 0.02  # 電極の太さ (mm)
Nx, Ny = 60, 150
dx = W / (Nx - 1)
dy = H / (Ny - 1)
C1=(dy**2)/(2*(dx**2+dy**2))
C2=(dx**2)/(2*(dx**2+dy**2))

# 保存ファイル名
save_file = "rect_solution_0.6mm.npz"

#保存しているファイルの削除
if os.path.exists(save_file):
    os.remove(save_file)  # ← これを追加して、保存ファイルを消す


# --- すでに計算済みファイルがあれば読み込み ---
if os.path.exists(save_file):
    print(">>> 計算済みデータを読み込み中...")
    data = np.load(save_file)
    phi = data['phi']
    Ex = data['Ex']
    Ey = data['Ey']
    E_mag = data['E_mag']
    X = data['X']
    Y = data['Y']
else:
    print(">>> データが見つかりません。計算を開始します...")


    # --- グリッド ---
    x = np.linspace(-W/2, W/2, Nx)
    y = np.linspace(-H/2, H/2, Ny)
    X, Y = np.meshgrid(x, y)

    # --- 初期電位 ---
    phi = np.zeros((Ny, Nx))

    # --- 電極位置とマスク ---
    left_x_min = -d/2 - e_length
    left_x_max = -d/2
    right_x_min = d/2
    right_x_max = d/2 + e_length

    electrode_mask = (
        ((X >= left_x_min) & (X <= left_x_max) & (np.abs(Y) < e_width/2)) |
        ((X >= right_x_min) & (X <= right_x_max) & (np.abs(Y) < e_width/2))
    )
    phi[(X >= left_x_min) & (X <= left_x_max) & (np.abs(Y) < e_width/2)] = 95
    phi[(X >= right_x_min) & (X <= right_x_max) & (np.abs(Y) < e_width/2)] = -95

    # --- ラプラス方程式 Neumann条件 ---
    def solve_laplace_neumann(phi, tol=1e-6, max_iter=10000):
        for it in range(max_iter):
            phi_old = phi.copy()
            for i in range(1, Ny - 1):
                for j in range(1, Nx - 1):
                    if not electrode_mask[i, j]:
                        phi[i, j] =  C2*(phi[i+1, j] + phi[i-1, j] )+C1*( phi[i, j+1] + phi[i, j-1])
            # Neumann境界条件
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            res = np.max(np.abs(phi - phi_old))
            if res < tol:
              print(f"Converged in {it} iterations, residual = {res:.2e}")
              return phi
        print("⚠ max_iter に到達（未収束の可能性あり）")
        return phi


            #if np.max(np.abs(phi - phi_old)) < tol:
                #print(f"Converged in {it} iterations.")
                #break
        #return phi

    phi = solve_laplace_neumann(phi)


    # --- 電場計算 ---
    # --- 電場計算 ---
    Ex = np.zeros_like(phi)
    Ey = np.zeros_like(phi)

    for i in range(1, Ny-1):
       for j in range(1, Nx-1):
           Ex[i, j] = -(phi[i, j+1] - phi[i, j-1]) / (2 * dx)
           Ey[i, j] = -(phi[i+1, j] - phi[i-1, j]) / (2 * dy)

    E_mag = np.sqrt(Ex**2 + Ey**2)

    # --- データ保存 ---
    np.savez(save_file, phi=phi, Ex=Ex, Ey=Ey, E_mag=E_mag, X=X, Y=Y)
    print(f">>> 計算結果を保存しました: {save_file}")

# --- 2D 電場強度マップ（等高線図） ---
plt.figure(figsize=(6, 5))
levels = np.arange(0, 1320, 120)# 明示的にレベルを定義
norm = mcolors.Normalize(vmin=0, vmax=10)
contour = plt.contourf(X, Y, E_mag, levels=levels, cmap='inferno')
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
plt.title(r'$|E|$ (V/mm)')
plt.colorbar(contour, label=r'$|E|$ (V/mm)')
plt.axis('equal')
plt.show()


E_max=np.max(E_mag)
print(f"電場強度マップの最大値: {E_max:.4f} V/mm")

# --- 興味ある領域の指定 ---
x_min, x_max = -0.1, 0.1  # mm
y_min, y_max = -0.25, -0.05   # mm

# --- 領域マスク作成 ---
region_mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)

# --- 領域内の電場平均値 ---
mean_E = np.nanmean(E_mag[region_mask])
print(f"領域({x_min}〜{x_max} mm, {y_min}〜{y_max} mm)の電場の平均値: {mean_E:.4f} V/mm")

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