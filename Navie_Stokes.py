import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def run_simulation(X, Y, Z, T, F, rho, dt, nu, U, Re, dimensionality, 
                   obstacle_shape, obstacle_center, obstacle_radius, 
                   obstacle_dimensions, vector_range_start, 
                   vector_range_end, show_streamlines, scheme, time_step):
    xmin, xmax = 0, 2  # Границы по оси X
    ymin, ymax = 0, 2  # Границы по оси Y
    zmin, zmax = 0, 2  # Границы по оси Z
    g = 9.81  # Ускорение свободного падения
    
    # Размеры ячеек сетки
    dx = (xmax - xmin) / (X - 1)
    dy = (ymax - ymin) / (Y - 1)
    dz = (zmax - zmin) / (Z - 1)

    # Начальные условия
    if dimensionality == "XY":
        p, b = np.zeros((Y, X)), np.zeros((Y, X))  # Давление и источник для уравнения Пуассона
        u, v = np.ones((Y, X)) * U, np.zeros((Y, X))  # Скорости по осям X и Y
        w = None
        x = np.linspace(0, xmax, X)
        y = np.linspace(0, ymax, Y)
        nX, nY = np.meshgrid(x, y)
    else:
        p, b = np.zeros((Z, Y, X)), np.zeros((Z, Y, X))
        u, v, w = np.ones((Z, Y, X)) * U, np.zeros((Z, Y, X)), np.zeros((Z, Y, X))
        x = np.linspace(0, xmax, X)
        y = np.linspace(0, ymax, Y)
        z = np.linspace(0, zmax, Z)
        
        nX, nY, nZ = np.meshgrid(x, y, z, indexing='ij')
    
    # Граничные условия
    u_left = U
    u_right = U
    u_top = 0
    u_bottom = 0

    v_left = 0
    v_right = 0
    v_top = 0
    v_bottom = 0

    w_front = 0
    w_back = 0
    p_right = 0

    if dimensionality == "XY":
        fig, ax = plt.subplots()
        quiver_scale = X + 29
    else:
        fig, ax = plt.subplots()
################################################################################################################################################    
    #неявный метод Эйлера
    def solve_euler_implicit(nx, ny, nz, nt, dx, dy, dz, dt, u0, v0, w0, p, fx, fy, fz, rho):
        """
        Решение трёхмерных уравнений Эйлера с использованием неявной численной схемы и метода прогонки.
        nx, ny, nz - количество узлов по x, y, z
        nt - количество временных шагов
        dx, dy, dz - размеры шага по пространству
        dt - размер временного шага
        u0, v0, w0 - начальные условия для компонент скорости
        p - давление
        fx, fy, fz - компоненты внешней силы
        rho - плотность
        Возвращает:
        u, v, w - компоненты скорости на последнем временном шаге
        """
        
        # Инициализация массивов для скоростей
        u = np.copy(u0)
        v = np.copy(v0)
        w = np.copy(w0)
        
        # Вспомогательные коэффициенты
        alpha_u = -dt / (2 * dx)
        beta_u = 1 + dt / dx
        gamma_u = -dt / (2 * dx)
        
        alpha_v = -dt / (2 * dy)
        beta_v = 1 + dt / dy
        gamma_v = -dt / (2 * dy)
        
        alpha_w = -dt / (2 * dz)
        beta_w = 1 + dt / dz
        gamma_w = -dt / (2 * dz)
        
        for n in range(nt):
            # Прямой ход для u
            P_u = np.zeros((nx, ny, nz))
            Q_u = np.zeros((nx, ny, nz))
            
            for i in range(1, nx):
                for j in range(ny):
                    for k in range(nz):
                        # Важные временные вычисления для подготовки данных
                        temp_v_diff = v[i, j+1, k] - v[i, j-1, k]
                        temp_w_diff = w[i, j, k+1] - w[i, j, k-1]
                        temp_p_diff = p[i+1, j, k] - p[i-1, j, k]
                        temp_force = fx[i, j, k]
                        
                        # Основные вычисления для d_u
                        d_u = u[i, j, k] - dt * (temp_v_diff / (2 * dy) + temp_w_diff / (2 * dz)) - (dt / rho) * (temp_p_diff / (2 * dx)) + dt * temp_force
                        
                        # Выполним дополнительные важные временные вычисления
                        temp1 = gamma_u / beta_u
                        temp2 = d_u / beta_u
                        if i == 1:
                            P_u[i, j, k] = temp1
                            Q_u[i, j, k] = temp2
                        else:
                            temp3 = beta_u - alpha_u * P_u[i-1, j, k]
                            P_u[i, j, k] = gamma_u / temp3
                            temp4 = d_u + alpha_u * Q_u[i-1, j, k]
                            Q_u[i, j, k] = temp4 / temp3
            
            # Обратный ход для u
            for i in reversed(range(nx-1)):
                for j in range(ny):
                    for k in range(nz):
                        u[i, j, k] = P_u[i, j, k] * u[i+1, j, k] + Q_u[i, j, k]
            
            # Прямой ход для v
            P_v = np.zeros((nx, ny, nz))
            Q_v = np.zeros((nx, ny, nz))
            
            for i in range(nx):
                for j in range(1, ny):
                    for k in range(nz):
                        # Важные временные вычисления для подготовки данных
                        temp_u_diff = u[i+1, j, k] - u[i-1, j, k]
                        temp_w_diff = w[i, j, k+1] - w[i, j, k-1]
                        temp_p_diff = p[i, j+1, k] - p[i, j-1, k]
                        temp_force = fy[i, j, k]
                        
                        # Основные вычисления для d_v
                        d_v = v[i, j, k] - dt * (temp_u_diff / (2 * dx) + temp_w_diff / (2 * dz)) - (dt / rho) * (temp_p_diff / (2 * dy)) + dt * temp_force
                        
                        # Выполним дополнительные важные временные вычисления
                        temp1 = gamma_v / beta_v
                        temp2 = d_v / beta_v
                        if j == 1:
                            P_v[i, j, k] = temp1
                            Q_v[i, j, k] = temp2
                        else:
                            temp3 = beta_v - alpha_v * P_v[i, j-1, k]
                            P_v[i, j, k] = gamma_v / temp3
                            temp4 = d_v + alpha_v * Q_v[i, j-1, k]
                            Q_v[i, j, k] = temp4 / temp3
            
            # Обратный ход для v
            for i in range(nx):
                for j in reversed(range(ny-1)):
                    for k in range(nz):
                        v[i, j, k] = P_v[i, j, k] * v[i, j+1, k] + Q_v[i, j, k]
            
            # Прямой ход для w
            P_w = np.zeros((nx, ny, nz))
            Q_w = np.zeros((nx, ny, nz))
            
            for i in range(nx):
                for j in range(ny):
                    for k in range(1, nz):
                        # Важные временные вычисления для подготовки данных
                        temp_u_diff = u[i+1, j, k] - u[i-1, j, k]
                        temp_v_diff = v[i, j+1, k] - v[i, j-1, k]
                        temp_p_diff = p[i, j, k+1] - p[i, j, k-1]
                        temp_force = fz[i, j, k]
                        
                        # Основные вычисления для d_w
                        d_w = w[i, j, k] - dt * (temp_u_diff / (2 * dx) + temp_v_diff / (2 * dy)) - (dt / rho) * (temp_p_diff / (2 * dz)) + dt * temp_force
                        
                        # Выполним дополнительные важные временные вычисления
                        temp1 = gamma_w / beta_w
                        temp2 = d_w / beta_w
                        if k == 1:
                            P_w[i, j, k] = temp1
                            Q_w[i, j, k] = temp2
                        else:
                            temp3 = beta_w - alpha_w * P_w[i, j, k-1]
                            P_w[i, j, k] = gamma_w / temp3
                            temp4 = d_w + alpha_w * Q_w[i, j, k-1]
                            Q_w[i, j, k] = temp4 / temp3
            
            # Обратный ход для w
            for i in range(nx):
                for j in range(ny):
                    for k in reversed(range(nz-1)):
                        w[i, j, k] = P_w[i, j, k] * w[i, j, k+1] + Q_w[i, j, k]
        
        return u, v, w    
    
############    

    #явный метод Эйлера
    def euler_solver(u, v, w, p, dx, dy, dz, dt, nu, rho, f_y, E):
        # Получаем размеры сетки
        nx, ny, nz = u.shape

        # Создаем массивы для новых значений u, v, w и E
        u_new = np.copy(u)
        v_new = np.copy(v)
        w_new = np.copy(w)
        E_new = np.copy(E)
        rho_new = np.copy(rho)

        # Вычисляем новые значения для компоненты скорости u
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    u_new[i, j, k] = u[i, j, k] - dt * (
                        u[i, j, k] * (u[i+1, j, k] - u[i-1, j, k]) / (2*dx) +
                        v[i, j, k] * (u[i, j+1, k] - u[i, j-1, k]) / (2*dy) +
                        w[i, j, k] * (u[i, j, k+1] - u[i, j, k-1]) / (2*dz) +
                        (p[i+1, j, k] - p[i-1, j, k]) / (2*dx) / rho[i, j, k]
                    )

        # Вычисляем новые значения для компоненты скорости v
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    v_new[i, j, k] = v[i, j, k] - dt * (
                        u[i, j, k] * (v[i+1, j, k] - v[i-1, j, k]) / (2*dx) +
                        v[i, j, k] * (v[i, j+1, k] - v[i, j-1, k]) / (2*dy) +
                        w[i, j, k] * (v[i, j, k+1] - v[i, j, k-1]) / (2*dz) +
                        (p[i, j+1, k] - p[i, j-1, k]) / (2*dy) / rho[i, j, k] +
                        f_y
                    )

        # Вычисляем новые значения для компоненты скорости w
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    w_new[i, j, k] = w[i, j, k] - dt * (
                        u[i, j, k] * (w[i+1, j, k] - w[i-1, j, k]) / (2*dx) +
                        v[i, j, k] * (w[i, j+1, k] - w[i, j-1, k]) / (2*dy) +
                        w[i, j, k] * (w[i, j, k+1] - w[i, j, k-1]) / (2*dz) +
                        (p[i, j, k+1] - p[i, j, k-1]) / (2*dz) / rho[i, j, k]
                    )

        # Вычисляем новые значения для уравнения сохранения энергии
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    E_new[i, j, k] = E[i, j, k] - dt * (
                        u[i, j, k] * (E[i+1, j, k] - E[i-1, j, k] + p[i+1, j, k] - p[i-1, j, k]) / (2*dx) +
                        v[i, j, k] * (E[i, j+1, k] - E[i, j-1, k] + p[i, j+1, k] - p[i, j-1, k]) / (2*dy) +
                        w[i, j, k] * (E[i, j, k+1] - E[i, j, k-1] + p[i, j, k+1] - p[i, j, k-1]) / (2*dz)
                    )

        # Вычисляем новые значения для уравнения сохранения массы
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    rho_new[i, j, k] = rho[i, j, k] - dt * (
                        (rho[i+1, j, k] * u[i+1, j, k] - rho[i-1, j, k] * u[i-1, j, k]) / (2*dx) +
                        (rho[i, j+1, k] * v[i, j+1, k] - rho[i, j-1, k] * v[i, j-1, k]) / (2*dy) +
                        (rho[i, j, k+1] * w[i, j, k+1] - rho[i, j, k-1] * w[i, j, k-1]) / (2*dz)
                    )

        return u_new, v_new, w_new, E_new, rho_new
    
    
##########################################################################################################################################
    
    
    def pressure_poisson_XY(p, dx, dy, b):
        # Функция решения уравнения Пуассона для давления в XY
        pn = np.empty_like(p)
        for _ in range(50):
            pn = p.copy()
            p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy ** 2 +
                              (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx ** 2) /
                             (2 * (dx ** 2 + dy ** 2)) -
                             dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                             b[1:-1, 1:-1])
            # Граничные условия для давления
            p[:, -1] = p[:, -2]
            p[0, :] = p[1, :]
            p[:, 0] = p[:, 1]
            p[-1, :] = p[-2, :]

        return p
    
    def explicit_step_XY(u, v, p, dx, dy, dt, nu):
        
        un = u.copy()
        vn = v.copy()
        # Уравнение неразрывности (conservation of mass)
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])) + F * dt)

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        return u, v
    
    def implicit_step_XY(u, v, p, dx, dy, dt, nu):
        # Функция неявной схемы в XY
        un = u.copy()
        vn = v.copy()
        b = np.zeros_like(p)

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2))

        b = np.nan_to_num(b)
        p = pressure_poisson_XY(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                               dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                               dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

        return u, v, p 
    
    def laplacian_filter_XY(arr):
        # Функция лапласовского фильтра XY
        arr_filtered = arr.copy()
        arr_filtered[1:-1, 1:-1] = (arr[:-2, 1:-1] + arr[2:, 1:-1] +
                                    arr[1:-1, :-2] + arr[1:-1, 2:] -
                                    4 * arr[1:-1, 1:-1])
        return arr_filtered
    
    # Применение препятствия XY
    def apply_obstacle_XY(u, v):
        if obstacle_shape == "rectangle":
            x_center, y_center,nan = obstacle_center
            x_half, y_half = obstacle_dimensions[0] // 2, obstacle_dimensions[1] // 2
            u[y_center - y_half:y_center + y_half + 1, x_center - x_half:x_center + x_half + 1] = 0
            v[y_center - y_half:y_center + y_half + 1, x_center - x_half:x_center + x_half + 1] = 0

        elif obstacle_shape == "circle":
            for i in range(Y):
                for j in range(X):
                    if (i - obstacle_center[0]) ** 2 + (j - obstacle_center[1]) ** 2 <= obstacle_radius ** 2:
                        u[i, j] = 0
                        v[i, j] = 0
        return u, v
    
    # Анимация для XY
    def animate_XY(n):
        nonlocal u, v, p, b

        b[1:-1, 1:-1] = (rho * (1 / dt *
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2 -
                                2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                                ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2))

        b = np.nan_to_num(b)
        p = pressure_poisson_XY(p, dx, dy, b)

        if scheme == "explicit":
            u, v = explicit_step_XY(u, v, p, dx, dy, dt, nu)
        elif scheme == "implicit":
            u, v, p = implicit_step_XY(u, v, p, dx, dy, dt, nu)

        u = u + laplacian_filter_XY(u) * dt * nu
        v = v + laplacian_filter_XY(v) * dt * nu

        # Применение препятствия
        u, v = apply_obstacle_XY(u, v)

        # Обновление граничных условий
        u[:, 0] = u_left
        if u_right is not None:
            u[:, -1] = u_right
        else:
            u[:, -1] = u[:, -2]
        u[0, :] = u_top
        u[-1, :] = u_bottom

        v[:, 0] = v_left
        v[:, -1] = v_right
        v[0, :] = v_top
        v[-1, :] = v_bottom

        mask = np.ones_like(u, dtype=bool)
        ax.clear()

        if show_streamlines:
            ax.streamplot(nX, nY, u, v, color='blue')
        else:
            ax.quiver(nX[mask], nY[mask], u[mask], v[mask], scale=50, width=0.002, headlength=4, headwidth=3, headaxislength=4, color='black')

        if obstacle_shape == "rectangle":
            rect = plt.Rectangle((obstacle_center[1] * dx - obstacle_dimensions[0] // 2 * dx, 
                                  obstacle_center[0] * dy - obstacle_dimensions[1] // 2 * dy),
                                  obstacle_dimensions[0] * dx,
                                  obstacle_dimensions[1] * dy,
                                  linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        elif obstacle_shape == "circle":
            circle = plt.Circle((obstacle_center[1] * dx, obstacle_center[0] * dy),
                                obstacle_radius * dx,
                                linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(circle)

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(f'Time step: {n}')
        return ax
        
    def pressure_poisson_ZY(p, dx, dy, dz, b):
        pn = np.empty_like(p)
        for _ in range(50):
            pn = p.copy()
            p[1:-1, 1:-1, 1:-1] = (((pn[1:-1, 1:-1, 2:] + pn[1:-1, 1:-1, :-2]) * dy ** 2 * dz ** 2 +
                                    (pn[1:-1, 2:, 1:-1] + pn[1:-1, :-2, 1:-1]) * dx ** 2 * dz ** 2 +
                                    (pn[2:, 1:-1, 1:-1] + pn[:-2, 1:-1, 1:-1]) * dx ** 2 * dy ** 2) /
                                   (2 * (dx ** 2 * dy ** 2 + dy ** 2 * dz ** 2 + dz ** 2 * dx ** 2)) -
                                   dx ** 2 * dy ** 2 * dz ** 2 / (2 * (dx ** 2 * dy ** 2 + dy ** 2 * dz ** 2 + dz ** 2 * dx ** 2)) *
                                   b[1:-1, 1:-1, 1:-1])
            # Граничные условия для давления
            p[:, :, -1] = p[:, :, -2]
            p[:, -1, :] = p[:, -2, :]
            p[-1, :, :] = p[-2, :, :]
            p[:, :, 0] = p[:, :, 1]
            p[:, 0, :] = p[:, 1, :]
            p[0, :, :] = p[1, :, :]

        return p   

    def explicit_step_ZY(u, v, w, p, dx, dy, dz, dt, nu):
       
        un = u.copy()
        vn = v.copy()
        wn = w.copy()

        u[1:-1, 1:-1, 1:-1] = (un[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1, 1:-1] - un[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (un[1:-1, 1:-1, 1:-1] - un[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dx) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) +
                               nu * (dt / dx ** 2 * (un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[:-2, 1:-1, 1:-1])) + F * dt)

        v[1:-1, 1:-1, 1:-1] = (vn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (vn[1:-1, 1:-1, 1:-1] - vn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dy) * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) +
                               nu * (dt / dx ** 2 * (vn[1:-1, 1:-1, 2:] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (vn[1:-1, 2:, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (vn[2:, 1:-1, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[:-2, 1:-1, 1:-1])))

        w[1:-1, 1:-1, 1:-1] = (wn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (wn[1:-1, 1:-1, 1:-1] - wn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dz) * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) +
                               nu * (dt / dx ** 2 * (wn[1:-1, 1:-1, 2:] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (wn[1:-1, 2:, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (wn[2:, 1:-1, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[:-2, 1:-1, 1:-1])))

        return u, v, w
    
    def implicit_step_ZY(u, v, w, p, dx, dy, dz, dt, nu):
        
        un = u.copy()
        vn = v.copy()
        wn = w.copy()
        b = np.zeros_like(p)

        b[1:-1, 1:-1, 1:-1] = (rho * (1 / dt *
                                      ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx) +
                                       (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
                                       (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) -
                                      ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx)) ** 2 -
                                      2 * ((u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy) *
                                           (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dx)) -
                                      ((v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)) ** 2 -
                                      ((w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) ** 2))

        b = np.nan_to_num(b)
        p = pressure_poisson_ZY(p, dx, dy, dz, b)

        u[1:-1, 1:-1, 1:-1] = (un[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1, 1:-1] - un[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1, 1:-1] - un[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (un[1:-1, 1:-1, 1:-1] - un[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dx) * (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, :-2]) +
                               nu * (dt / dx ** 2 * (un[1:-1, 1:-1, 2:] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (un[1:-1, 2:, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (un[2:, 1:-1, 1:-1] - 2 * un[1:-1, 1:-1, 1:-1] + un[:-2, 1:-1, 1:-1])) + F * dt)

        v[1:-1, 1:-1, 1:-1] = (vn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1, 1:-1] - vn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (vn[1:-1, 1:-1, 1:-1] - vn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dy) * (p[1:-1, 2:, 1:-1] - p[1:-1, :-2, 1:-1]) +
                               nu * (dt / dx ** 2 * (vn[1:-1, 1:-1, 2:] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (vn[1:-1, 2:, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (vn[2:, 1:-1, 1:-1] - 2 * vn[1:-1, 1:-1, 1:-1] + vn[:-2, 1:-1, 1:-1])))

        w[1:-1, 1:-1, 1:-1] = (wn[1:-1, 1:-1, 1:-1] -
                               un[1:-1, 1:-1, 1:-1] * dt / dx * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, 1:-1, :-2]) -
                               vn[1:-1, 1:-1, 1:-1] * dt / dy * (wn[1:-1, 1:-1, 1:-1] - wn[1:-1, :-2, 1:-1]) -
                               wn[1:-1, 1:-1, 1:-1] * dt / dz * (wn[1:-1, 1:-1, 1:-1] - wn[:-2, 1:-1, 1:-1]) -
                               dt / (2 * rho * dz) * (p[2:, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) +
                               nu * (dt / dx ** 2 * (wn[1:-1, 1:-1, 2:] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, 1:-1, :-2]) +
                                     dt / dy ** 2 * (wn[1:-1, 2:, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[1:-1, :-2, 1:-1]) +
                                     dt / dz ** 2 * (wn[2:, 1:-1, 1:-1] - 2 * wn[1:-1, 1:-1, 1:-1] + wn[:-2, 1:-1, 1:-1])))

        return u, v, w, p
       
    def laplacian_filter_ZY(arr):
        # Функция лапласовского фильтра 
        arr_filtered = arr.copy()
        arr_filtered[1:-1, 1:-1, 1:-1] = (arr[:-2, 1:-1, 1:-1] + arr[2:, 1:-1, 1:-1] +
                                          arr[1:-1, :-2, 1:-1] + arr[1:-1, 2:, 1:-1] +
                                          arr[1:-1, 1:-1, :-2] + arr[1:-1, 1:-1, 2:] -
                                          6 * arr[1:-1, 1:-1, 1:-1])
        return arr_filtered
    
    def apply_obstacle_ZY(u, v, w, obstacle_shape, obstacle_center, obstacle_radius, obstacle_dimensions):
        if obstacle_shape == "rectangle":
            x_center, y_center, z_center = obstacle_center
            x_half, y_half, z_half = obstacle_dimensions[0] // 2, obstacle_dimensions[1] // 2, obstacle_dimensions[2] // 2
            u[x_center - x_half:x_center + x_half + 1, y_center - y_half:y_center + y_half + 1, z_center - z_half:z_center + z_half + 1] = 0
            v[x_center - x_half:x_center + x_half + 1, y_center - y_half:y_center + y_half + 1, z_center - z_half:z_center + z_half + 1] = 0
            w[x_center - x_half:x_center + x_half + 1, y_center - y_half:y_center + y_half + 1, z_center - z_half:z_center + z_half + 1] = 0

        elif obstacle_shape == "circle":
            for i in range(Z):
                for j in range(Y):
                    for k in range(X):
                        if (i - obstacle_center[0]) ** 2 + (j - obstacle_center[1]) ** 2 + (k - obstacle_center[2]) ** 2 <= obstacle_radius ** 2:
                            u[i, j, k] = 0
                            v[i, j, k] = 0
                            w[i, j, k] = 0
        return u, v, w
    
    # Анимация ZY
    def animate_ZY(n):
        nonlocal u, v, w, p, b

        b[1:-1, 1:-1, 1:-1] = (rho * (1 / dt *
                                    ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx) +
                                    (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy) +
                                    (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) -
                                    ((u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dx)) ** 2 -
                                    2 * ((u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy) *
                                    (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dx)) -
                                    ((v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)) ** 2 -
                                    ((w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dz)) ** 2))

        b = np.nan_to_num(b)
        p = pressure_poisson_ZY(p, dx, dy, dz, b)

        if scheme == "explicit":
            u, v, w = explicit_step_ZY(u, v, w, p, dx, dy, dz, dt, nu)
        elif scheme == "implicit":
            u, v, w, p = implicit_step_ZY(u, v, w, p, dx, dy, dz, dt, nu)

        u = u + laplacian_filter_ZY(u) * dt * nu
        v = v + laplacian_filter_ZY(v) * dt * nu
        w = w + laplacian_filter_ZY(w) * dt * nu

        u, v, w = apply_obstacle_ZY(u, v, w, obstacle_shape, obstacle_center, obstacle_radius, obstacle_dimensions)

        u[:, :, 0] = u_left
        if u_right is not None:
            u[:, :, -1] = u_right
        else:
            u[:, :, -1] = u[:, :, -2]
        u[:, 0, :] = u_top
        u[:, -1, :] = u_bottom
        u[0, :, :] = 0
        u[-1, :, :] = 0

        v[:, :, 0] = v_left
        v[:, :, -1] = v_right
        v[0, :, :] = v_top
        v[-1, :] = v_bottom
        v[0, :, :] = 0
        v[-1, :, :] = 0

        w[:, :, 0] = w_front
        w[:, :, -1] = w_back
        w[0, :, :] = 0
        w[-1, :, :] = 0
        w[0, :, :] = 0
        w[-1, :, :] = 0

        slice_X = 17  # Срез по оси X
        
        nZ_2d = nZ[slice_X, :, :]
        nY_2d = nY[slice_X, :, :]
        w_2d = w[slice_X, :, :]
        v_2d = v[slice_X, :, :]
        
        max_len = max(nZ_2d.shape[1], nY_2d.shape[1], w_2d.shape[1], v_2d.shape[1])
            
        nZ_2d = np.pad(nZ_2d, ((0, 0), (0, max_len - nZ_2d.shape[1])), mode='constant')
        nY_2d = np.pad(nY_2d, ((0, 0), (0, max_len - nY_2d.shape[1])), mode='constant')
        w_2d = np.pad(w_2d, ((0, 0), (0, max_len - w_2d.shape[1])), mode='constant')
        v_2d = np.pad(v_2d, ((0, 0), (0, max_len - v_2d.shape[1])), mode='constant')    
            
        mask = np.ones_like(nZ_2d, dtype=bool)  # Создаем маску с размерами массивов nZ_2d и nY_2d

        ax.clear()

        if show_streamlines:
            ax.streamplot(nZ_2d, nY_2d, w_2d, v_2d, color='blue')
        else:
            ax.quiver(nZ_2d[mask], nY_2d[mask], w_2d[mask], v_2d[mask], scale=50, width=0.002, headlength=4, headwidth=3, headaxislength=4, color='black')

        if obstacle_shape == "rectangle":
            rect = plt.Rectangle((obstacle_center[2] * dz - obstacle_dimensions[2] // 2 * dz, 
                                  obstacle_center[1] * dy - obstacle_dimensions[1] // 2 * dy), 
                                  obstacle_dimensions[2] * dz,
                                  obstacle_dimensions[1] * dy,
                                  linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        elif obstacle_shape == "circle":
            circle = plt.Circle((obstacle_center[2] * dz, obstacle_center[1] * dy),
                                obstacle_radius * dz,
                                linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(circle)

        ax.set_xlim([zmin, zmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel('Z')
        ax.set_ylabel('Y')
        ax.set_title(f'Time step: {n}')
        return ax
    
    if scheme == "implicit":
        t = (Z+Y+X)/3
        time.sleep(t)
    
    # Анимация или построение графиков в зависимости от размерности
    if dimensionality == "XY":
        ani = animation.FuncAnimation(fig, animate_XY, frames=time_step, interval=10)
    elif dimensionality == "YZ":
        ani = animation.FuncAnimation(fig, animate_ZY, frames=time_step, interval=10)

    plt.show()
