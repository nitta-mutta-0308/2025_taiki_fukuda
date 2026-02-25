# --- 12/3時点の菱形流路（大） 安全版 ---
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from mpl_toolkits.mplot3d import Axes3D

# --- 菱形パラメータ ---
d_h, d_v = 0.9, 1.5  # 横・縦の対角線長 (mm)
a, b = d_h / 2, d_v / 2

# --- 頂点（左,上,右,下） ---
x0, y0 = -a, 0
x1, y1 = 0, b
x2, y2 = a, 0
x3, y3 = 0, -b

# --- 基底ベクトル ---
#v1 = np.array([x1 - x0, y1 - y0])#v1=[0.3,0.75]
#v2 = np.array([x3 - x0, y3 - y0])#v1=[0.3,-0.75]

# --- 保存ファイル名 ---
save_file = "diamond_solution_big.npz"

#保存しているファイルの削除
if os.path.exists(save_file):
    os.remove(save_file)  # ← これを追加して、保存ファイルを消す


# --- すでに計算済みファイルがあれば読み込み ---
if os.path.exists(save_file):
    print(">>> 計算済みデータを読み込み中...")
    data = np.load(save_file)
    phi = data['phi']
    Eu=data['Eu']
    Ev=data['Ev']
    Ex = data['Ex']
    Ey = data['Ey']
    E_mag = data['E_mag']
    X = data['X']
    Y = data['Y']
else:
    print(">>> データが見つかりません。計算を開始します...")

    # --- UVパラメトリックグリッド ---
    Nu, Nv = 87, 87
    u = np.linspace(0, 1, Nu)
    v = np.linspace(0, 1, Nv)#v 方向を「Nv 個の格子点」に分割している
    U, V = np.meshgrid(u, v, indexing='ij')#u = [u0, u1, u2, ..., u_{Nu-1}]v = [v0, v1, v2, ..., v_{Nv-1}]

    # --- 回転・拡大縮小行列 ---
    theta = np.pi/4  # 45度
    sx, sy = 0.9/np.sqrt(2), 1.5/np.sqrt(2)
    R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
    S = np.array([[sx, 0],[0, sy]])

    J = S@R   # 拡大縮小 × 回転

    # --- XY変換 ---
    X =  U * J[0,0]+ V * J[0,1]
    Y =  U * J[1,0] + V * J[1,1]-0.75


    du = u[1] - u[0]
    dv = v[1] - v[0]

    # --- 初期電位 ---
    phi = np.zeros_like(U)#phi = np.zeros((Nu, Nv))と同義

    # --- 電極パラメータ ---
    d = 0.2
    e_length = 0.35
    e_width = 0.02

    left_x_min = -d/2 - e_length
    left_x_max = -d/2
    right_x_min = d/2
    right_x_max = d/2 + e_length

    diamond_mask=(np.abs(X)/a + np.abs(Y)/b) <= 1
    # --- 電極マスク ---
    electrode_mask = (
        ((X >= left_x_min) & (X <= left_x_max) & (np.abs(Y) < e_width/2)) |
        ((X >= right_x_min) & (X <= right_x_max) & (np.abs(Y) < e_width/2))
    )& diamond_mask

    # --- Dirichlet 電位設定 ---
    phi[electrode_mask & (X <= left_x_max)] = 95.0#X(u,v)Y(u,v)で、X,Yの対応するグリッドはuvにも反映される
    phi[electrode_mask & (X >= right_x_min)] = -95.0

    # --- Laplace方程式 Jacobi法 + Neumann境界 ---
    def solve_laplace_uv(phi, electrode_mask, tol=1e-6, max_iter=20000):
        Nu, Nv = phi.shape
        for it in range(max_iter):
            phi_old = phi.copy()
            for i in range(1, Nu-1):
                for j in range(1, Nv-1):
                    if electrode_mask[i,j]:
                        continue
                    phi[i,j] = 0.25*(phi_old[i+1,j]+phi_old[i-1,j]+phi_old[i,j+1]+phi_old[i,j-1])
            # Neumann境界条件
            phi[0,:] = phi[1,:]
            phi[-1,:] = phi[-2,:]
            phi[:,0] = phi[:,1]
            phi[:,-1] = phi[:,-2]
            # 電極条件再設定
            phi[electrode_mask & (X <= left_x_max)] = 95.0
            phi[electrode_mask & (X >= right_x_min)] = -95.0

            diff = np.max(np.abs(phi - phi_old))
            if diff < tol:
                print(f">>> 収束: {it}回 (Δφ={diff:.2e})")
                break
        return phi

    print(">>> UV空間で電位を解きます...")
    phi = solve_laplace_uv(phi, electrode_mask)

    #電場計算
    Eu=np.zeros_like(phi)
    Ev=np.zeros_like(phi)

    for i in range(1,Nu-1):
      for j in range(1,Nv-1):
        Eu[i,j]=-(phi[i+1,j]-phi[i-1,j])/(2*du)
        Ev[i,j]=-(phi[i,j+1]-phi[i,j-1])/(2*dv)

    # --- 平滑化 ---
    #phi_smoothed = gaussian_filter(phi, sigma=1)

    # --- 電場計算 ---
    #phi_u, phi_v = np.gradient(phi, du, dv, edge_order=2)
    JinvT = np.linalg.inv(J).T  #okここまでJの逆行列の置換

    Ex=Ev*JinvT[0,0]+Eu*JinvT[0,1]
    Ey=Ev*JinvT[1,0]+Eu*JinvT[1,1]



    #grad_uv = np.vstack([phi_u.flatten(), phi_v.flatten()])
    #grad_xy = JinvT @ grad_uv

    #Ex = -grad_xy[0,:].reshape(Nu,Nv)
    #Ey = -grad_xy[1,:].reshape(Nu,Nv)
    E_mag = np.sqrt(Ex**2 + Ey**2)

    # --- データ保存 ---
    np.savez(save_file, phi=phi,Eu=Eu,Ev=Ev, Ex=Ex, Ey=Ey, E_mag=E_mag, X=X, Y=Y)
    print(f">>> 計算結果を保存しました: {save_file}")

# --- 可視化 ---
Xv = np.array([-a,0,a,0,-a])
Yv = np.array([0,b,0,-b,0])



# 電場強度マップ
plt.figure(figsize=(6,5))
levels=np.arange(0,1320,120)
contour = plt.contourf(X, Y, E_mag, levels=levels, cmap='inferno')
plt.plot(Xv,Yv,'k',lw=2)
plt.xlabel(r'$x$ (mm)')
plt.ylabel(r'$y$ (mm)')
plt.title(r'$|E|$ (V/mm)')
plt.colorbar(contour, label=r'$|E|$ (V/mm)')
plt.axis('equal')
plt.show()


# 最大値
print('max |E| =', np.max(E_mag))
# --- 興味領域の平均電場 ---
x_min, x_max = -0.1, 0.1
y_min, y_max = -0.25, -0.05
region_mask = (X>=x_min)&(X<=x_max)&(Y>=y_min)&(Y<=y_max)
mean_E = np.nanmean(E_mag[region_mask])
print(f"領域({x_min}〜{x_max} mm, {y_min}〜{y_max} mm)の平均電場: {mean_E:.4f} V/mm")

# --- 領域内の電場平均値 ---
left_mask  = X < 0
right_mask = X > 0
E_max_left  = np.max(E_mag[left_mask])
E_max_right = np.max(E_mag[right_mask])

print(f"x<0 側の最大電場: {E_max_left:.6f} V/mm")
print(f"x>0 側の最大電場: {E_max_right:.6f} V/mm")

plt.figure(figsize=(5,5))
levels = np.linspace(np.min(phi), np.max(phi), 50)

plt.contourf(U, V, phi, levels=levels, cmap='inferno')
plt.colorbar(label='φ (V)')
plt.xlabel('u')
plt.ylabel('v')
plt.title('Potential distribution in uv space')
plt.axis('equal')
plt.show()


# --- 興味領域の平均電場 ---
x_min, x_max = -0.1, 0.1
y_min, y_max = -0.22, -0.02
region_mask = (X>=x_min)&(X<=x_max)&(Y>=y_min)&(Y<=y_max)
mean_E = np.nanmean(E_mag[region_mask])
print(f"領域({x_min}〜{x_max} mm, {y_min}〜{y_max} mm)の平均電場: {mean_E:.4f} V/mm")