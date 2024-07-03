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
