import tkinter as tk
import time
from tkinter import ttk, messagebox
from Navie_Stokes import run_simulation
import time

# Параметры симуляции (значения по умолчанию)
X, Y, Z = 30, 30, 30  # Размеры сетки по координатам
T = 1000  # Количество временных шагов
F = 2  # Внешняя сила
rho = 0.1  # Плотность жидкости
dt = 0.0009  # Временной шаг
nu = 0.1
U = 1  # Скорость потока
Re = 10  # Число Рейнольдса
dimensionality = "XY"
obstacle_shape = "circle"
obstacle_center = (15, 15, 15)
obstacle_radius = 3
obstacle_dimensions = (10, 10, 10)
vector_range_start = 1
vector_range_end = 3
show_streamlines = False
scheme = "explicit"
time_step = 500

class RoundedEntry(tk.Canvas):
    def __init__(self, parent, *args, **kwargs):
        tk.Canvas.__init__(self, parent, height=30, width=200, bg="white", highlightthickness=0, relief='flat')
        self.rounded_rect(5, 5, 195, 25, 10, outline="#d9d9d9", width=1)
        self.entry = tk.Entry(self, *args, **kwargs, bd=0, highlightthickness=0, font=('Helvetica', 12))
        self.create_window(100, 15, window=self.entry)
        
    def rounded_rect(self, x1, y1, x2, y2, radius=25, **kwargs):
        points = [x1+radius, y1,
                  x1+radius, y1,
                  x2-radius, y1,
                  x2-radius, y1,
                  x2, y1,
                  x2, y1+radius,
                  x2, y1+radius,
                  x2, y2-radius,
                  x2, y2-radius,
                  x2, y2,
                  x2-radius, y2,
                  x2-radius, y2,
                  x1+radius, y2,
                  x1+radius, y2,
                  x1, y2,
                  x1, y2-radius,
                  x1, y2-radius,
                  x1, y1+radius,
                  x1, y1]
        return self.create_polygon(points, **kwargs, smooth=True)

def update_obstacle_fields():
    if obstacle_var.get() == "rectangle":
        obstacle_center_frame.grid()
        obstacle_radius_frame.grid_remove()
        obstacle_dimensions_frame.grid()
    elif obstacle_var.get() == "circle":
        obstacle_center_frame.grid()
        obstacle_radius_frame.grid()
        obstacle_dimensions_frame.grid_remove()

def start_simulation():
    try:
        X = int(entry_X.entry.get())
        Y = int(entry_Y.entry.get())
        Z = int(entry_Z.entry.get())
        dimensionality = dim_var.get()
        obstacle_shape = obstacle_var.get()

        if X <= 0 or Y <= 0 or Z <= 0:
            raise ValueError("Размеры сетки должны быть положительными числами.")
        if dimensionality not in ["XY", "YZ"]:
            raise ValueError("Некорректное значение для размерности.")
        if obstacle_shape not in ["rectangle", "circle"]:
            raise ValueError("Некорректное значение для формы препятствия.")

        if obstacle_shape == "rectangle":
            obstacle_center = (int(entry_obstacle_center_x.entry.get()), int(entry_obstacle_center_y.entry.get()), int(entry_obstacle_center_z.entry.get()))
            obstacle_dimensions = (int(entry_obstacle_width.entry.get()), int(entry_obstacle_height.entry.get()), int(entry_obstacle_depth.entry.get()))
            obstacle_radius = None

            if any(dim <= 0 for dim in obstacle_dimensions):
                raise ValueError("Размеры прямоугольного препятствия должны быть положительными числами.")
        elif obstacle_shape == "circle":
            obstacle_center = (int(entry_obstacle_center_x.entry.get()), int(entry_obstacle_center_y.entry.get()), int(entry_obstacle_center_z.entry.get()))
            obstacle_radius = int(entry_obstacle_radius.entry.get())
            obstacle_dimensions = None

            if obstacle_radius <= 0:
                raise ValueError("Радиус круглого препятствия должен быть положительным числом.")

        show_streamlines = bool(show_streamlines_var.get())
        scheme = scheme_var.get()

        print(f"X: {X}, Y: {Y}, Z: {Z}")
        print(f"dimensionality: {dimensionality}, obstacle_shape: {obstacle_shape}")
        print(f"obstacle_center: {obstacle_center}, obstacle_radius: {obstacle_radius}")
        print(f"obstacle_dimensions: {obstacle_dimensions}")
        print(f"show_streamlines: {show_streamlines}, scheme: {scheme}, time_step: {time_step}")
        xIsEnabled= False
        yIsEnabled= False
        zIsEnabled= False
        time.sleep(1)
        if obstacle_shape == "circle":
            xIsEnabled = X/obstacle_radius<=7
            yIsEnabled = Y/obstacle_radius<=7
            zIsEnabled = Z/obstacle_radius<=7
      
        
        if xIsEnabled and yIsEnabled and zIsEnabled:
            raise ValueError("Размеры препятствия не должны быть в 7 раз меньше сетки.")
            return
        
        run_simulation(X, Y, Z, T, F, rho, dt, nu, U, Re, dimensionality, 
                       obstacle_shape, obstacle_center, obstacle_radius, 
                       obstacle_dimensions, vector_range_start, 
                       vector_range_end, show_streamlines, scheme, time_step)

    except ValueError as e:
        messagebox.showerror("Ошибка ввода", str(e))

root = tk.Tk()
root.title("Параметры симуляции")

# Apply styles
style = ttk.Style()
style.configure('TFrame', background='#e0f7fa')
style.configure('TLabel', background='#e0f7fa', font=('Helvetica', 12), foreground='#00796b')
style.configure('TButton', background='#004d40', foreground='#00796b', font=('Helvetica', 12, 'bold'))
style.configure('TRadiobutton', background='#e0f7fa', font=('Helvetica', 12), foreground='#00796b')
style.configure('TCheckbutton', background='#e0f7fa', font=('Helvetica', 12), foreground='#00796b')

mainframe = ttk.Frame(root, padding="10 10 10 10")
mainframe.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Label(mainframe, text="Количество узлов по X:").grid(column=1, row=1, sticky=tk.W)
entry_X = RoundedEntry(mainframe)
entry_X.grid(column=2, row=1, sticky=(tk.W, tk.E))
entry_X.entry.insert(0, str(X))

ttk.Label(mainframe, text="Количество узлов по Y:").grid(column=1, row=2, sticky=tk.W)
entry_Y = RoundedEntry(mainframe)
entry_Y.grid(column=2, row=2, sticky=(tk.W, tk.E))
entry_Y.entry.insert(0, str(Y))

ttk.Label(mainframe, text="Количество узлов по Z:").grid(column=1, row=3, sticky=tk.W)
entry_Z = RoundedEntry(mainframe)
entry_Z.grid(column=2, row=3, sticky=(tk.W, tk.E))
entry_Z.entry.insert(0, str(Z))

dim_var = tk.StringVar()
dim_var.set(dimensionality)
ttk.Label(mainframe, text="Срезы по осям:").grid(column=1, row=4, sticky=tk.W)
ttk.Radiobutton(mainframe, text="xy", variable=dim_var, value="XY").grid(column=2, row=4, sticky=tk.W)
ttk.Radiobutton(mainframe, text="zy", variable=dim_var, value="YZ").grid(column=3, row=4, sticky=tk.W)
ttk.Radiobutton(mainframe, text="3D", variable=dim_var, value="3D").grid(column=3, row=4, sticky=tk.W)

obstacle_var = tk.StringVar()
obstacle_var.set(obstacle_shape)
ttk.Label(mainframe, text="Форма препятствия:").grid(column=1, row=5, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Прямоугольник", variable=obstacle_var, value="rectangle", command=update_obstacle_fields).grid(column=2, row=5, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Круг", variable=obstacle_var, value="circle", command=update_obstacle_fields).grid(column=3, row=5, sticky=tk.W)

obstacle_center_frame = ttk.Frame(mainframe)
obstacle_center_frame.grid(column=1, row=6, columnspan=3, sticky=(tk.W, tk.E))
ttk.Label(obstacle_center_frame, text="Центр препятствия X:").grid(column=1, row=1, sticky=tk.W)
entry_obstacle_center_x = RoundedEntry(obstacle_center_frame)
entry_obstacle_center_x.grid(column=2, row=1, sticky=(tk.W, tk.E))
entry_obstacle_center_x.entry.insert(0, str(obstacle_center[0]))

ttk.Label(obstacle_center_frame, text="Центр препятствия Y:").grid(column=1, row=2, sticky=tk.W)
entry_obstacle_center_y = RoundedEntry(obstacle_center_frame)
entry_obstacle_center_y.grid(column=2, row=2, sticky=(tk.W, tk.E))
entry_obstacle_center_y.entry.insert(0, str(obstacle_center[1]))

ttk.Label(obstacle_center_frame, text="Центр препятствия Z:").grid(column=1, row=3, sticky=tk.W)
entry_obstacle_center_z = RoundedEntry(obstacle_center_frame)
entry_obstacle_center_z.grid(column=2, row=3, sticky=(tk.W, tk.E))
entry_obstacle_center_z.entry.insert(0, str(obstacle_center[2]))

obstacle_radius_frame = ttk.Frame(mainframe)
obstacle_radius_frame.grid(column=1, row=7, columnspan=3, sticky=(tk.W, tk.E))
ttk.Label(obstacle_radius_frame, text="Радиус препятствия:").grid(column=1, row=1, sticky=tk.W)
entry_obstacle_radius = RoundedEntry(obstacle_radius_frame)
entry_obstacle_radius.grid(column=2, row=1, sticky=(tk.W, tk.E))
entry_obstacle_radius.entry.insert(0, str(obstacle_radius))

obstacle_dimensions_frame = ttk.Frame(mainframe)
obstacle_dimensions_frame.grid(column=1, row=8, columnspan=3, sticky=(tk.W, tk.E))
ttk.Label(obstacle_dimensions_frame, text="Ширина (для прямоугольника):").grid(column=1, row=1, sticky=tk.W)
entry_obstacle_width = RoundedEntry(obstacle_dimensions_frame)
entry_obstacle_width.grid(column=2, row=1, sticky=(tk.W, tk.E))
entry_obstacle_width.entry.insert(0, str(obstacle_dimensions[0]))

ttk.Label(obstacle_dimensions_frame, text="Высота (для прямоугольника):").grid(column=1, row=2, sticky=tk.W)
entry_obstacle_height = RoundedEntry(obstacle_dimensions_frame)
entry_obstacle_height.grid(column=2, row=2, sticky=(tk.W, tk.E))
entry_obstacle_height.entry.insert(0, str(obstacle_dimensions[1]))

ttk.Label(obstacle_dimensions_frame, text="Глубина (для прямоугольника):").grid(column=1, row=3, sticky=tk.W)
entry_obstacle_depth = RoundedEntry(obstacle_dimensions_frame)
entry_obstacle_depth.grid(column=2, row=3, sticky=(tk.W, tk.E))
entry_obstacle_depth.entry.insert(0, str(obstacle_dimensions[2]))

show_streamlines_frame = ttk.Frame(mainframe)
show_streamlines_frame.grid(column=1, row=9, columnspan=3, sticky=(tk.W, tk.E))
show_streamlines_var = tk.IntVar()
show_streamlines_var.set(int(show_streamlines))
ttk.Checkbutton(show_streamlines_frame, text="Показать линии тока", variable=show_streamlines_var).grid(column=1, row=1, sticky=tk.W)

scheme_var = tk.StringVar()
scheme_var.set(scheme)
ttk.Label(mainframe, text="Схема:").grid(column=1, row=10, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Явная", variable=scheme_var, value="explicit").grid(column=2, row=10, sticky=tk.W)
ttk.Radiobutton(mainframe, text="Неявная", variable=scheme_var, value="implicit").grid(column=3, row=10, sticky=tk.W)

ttk.Button(mainframe, text="Начать симуляцию", command=start_simulation).grid(column=2, row=12, sticky=tk.W)

for child in mainframe.winfo_children():
    child.grid_configure(padx=5, pady=5)

update_obstacle_fields()


def runGui():
    root.mainloop()

runGui()
