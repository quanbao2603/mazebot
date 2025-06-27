
import customtkinter as ctk
import tkinter as tk
import logic as logic
import time

ctk.set_appearance_mode('light')
ctk.set_default_color_theme('green')

ALGORITHMS = ['BFS', 'DFS', 'Dijkstra', 'A*']
COLORS = {
    'BFS': {'visit': '#FFA500', 'path': '#FF4500'},
    'DFS': {'visit': '#00CED1', 'path': '#008B8B'},
    'Dijkstra': {'visit': '#9370DB', 'path': '#4B0082'},
    'A*': {'visit': '#4682B4', 'path': '#00008B'}
}

class MazeCompareApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('So sánh thuật toán tìm đường')
        self.state('zoomed')
        self.grid_size = 30
        self.grid_data = None
        self.start = (0, 0)
        self.end = (self.grid_size-1, self.grid_size-1)
        self.delay = 30
        self.metrics = [{}, {}]
        self.running = False

        # --- Panel chọn thuật toán và điều khiển ---
        control = ctk.CTkFrame(self, width=320, fg_color='#00a000')
        control.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        ctk.CTkLabel(control, text='So sánh hai thuật toán', font=('Arial', 20, 'bold'), text_color='white').pack(pady=(20,10))
        ctk.CTkLabel(control, text='Thuật toán 1', text_color='white').pack(pady=(10,2))
        self.combo1 = ctk.CTkComboBox(control, values=ALGORITHMS, width=260)
        self.combo1.set('BFS')
        self.combo1.pack(pady=(0,10))
        ctk.CTkLabel(control, text='Thuật toán 2', text_color='white').pack(pady=(10,2))
        self.combo2 = ctk.CTkComboBox(control, values=ALGORITHMS, width=260)
        self.combo2.set('A*')
        self.combo2.pack(pady=(0,10))
        ctk.CTkLabel(control, text='Kích thước lưới', text_color='white').pack(pady=(10,2))
        self.size_slider = ctk.CTkSlider(control, from_=10, to=60, number_of_steps=50, command=self.update_size)
        self.size_slider.set(self.grid_size)
        self.size_slider.pack(pady=(0,5))
        self.size_label = ctk.CTkLabel(control, text=str(self.grid_size), text_color='white')
        self.size_label.pack()
        self.btn_maze = ctk.CTkButton(control, text='Tạo mê cung', command=self.generate_maze, fg_color='white', text_color='black', hover_color='red', width=260)
        self.btn_maze.pack(pady=(20,5))
        self.btn_compare = ctk.CTkButton(control, text='So sánh', command=self.start_compare, fg_color='white', text_color='black', hover_color='red', width=260)
        self.btn_compare.pack(pady=(5,20))
        # --- Bảng số liệu ---
        self.metrics_frame = ctk.CTkFrame(control, fg_color='#004d00', corner_radius=10)
        self.metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        self.metric_labels = []
        for i, name in enumerate(['Thuật toán 1', 'Thuật toán 2']):
            ctk.CTkLabel(self.metrics_frame, text=name, text_color='white', font=('Arial', 14, 'bold')).grid(row=0, column=i+1, padx=10)
        for idx, metric in enumerate(['Visited', 'Path length', 'Time (s)']):
            ctk.CTkLabel(self.metrics_frame, text=metric+':', text_color='white').grid(row=idx+1, column=0, sticky='w', padx=5)
            row_labels = []
            for i in range(2):
                lbl = ctk.CTkLabel(self.metrics_frame, text='0', text_color='white')
                lbl.grid(row=idx+1, column=i+1, padx=10)
                row_labels.append(lbl)
            self.metric_labels.append(row_labels)

        # --- Hai canvas lưới ---
        main_frame = ctk.CTkFrame(self, fg_color='white')
        main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas1 = tk.Canvas(main_frame, bg='white')
        self.canvas2 = tk.Canvas(main_frame, bg='white')
        self.canvas1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas1.bind('<Configure>', lambda e: self.draw_grid(self.canvas1, 0))
        self.canvas2.bind('<Configure>', lambda e: self.draw_grid(self.canvas2, 1))
        self.cell_size = [None, None]
        self.pad_x = [None, None]
        self.pad_y = [None, None]
        self.cells = [{}, {}]
        self.generators = [None, None]
        self.start_end = [self.start, self.end]
        self.after_id = None
        self.generate_maze()

    def update_size(self, val):
        self.grid_size = int(val)
        self.size_label.configure(text=str(self.grid_size))
        self.generate_maze()

    def find_nearest_empty(self, grid, pos):
        from collections import deque
        n = len(grid)
        visited = set()
        queue = deque([pos])
        while queue:
            r, c = queue.popleft()
            if grid[r][c] == 0:
                return (r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return pos

    def generate_maze(self):
        n = self.grid_size
        self.grid_data = logic.generate_maze(n, 'Recursive Backtracking')
        self.start = self.find_nearest_empty(self.grid_data, (0, 0))
        self.end = self.find_nearest_empty(self.grid_data, (n-1, n-1))
        self.start_end = [self.start, self.end]
        for i in range(2):
            self.draw_grid(self.canvas1 if i==0 else self.canvas2, i)
        for row_labels in self.metric_labels:
            for lbl in row_labels:
                lbl.configure(text='0')

    def draw_grid(self, canvas, idx):
        n = self.grid_size
        canvas.delete('all')
        w, h = canvas.winfo_width(), canvas.winfo_height()
        cell = min(w / n, h / n)
        total = cell * n
        pad_x = (w - total) / 2
        pad_y = (h - total) / 2
        self.cell_size[idx] = cell
        self.pad_x[idx] = pad_x
        self.pad_y[idx] = pad_y
        self.cells[idx] = {}
        for row in range(n):
            for col in range(n):
                x1 = pad_x + col * cell
                y1 = pad_y + row * cell
                x2 = x1 + cell
                y2 = y1 + cell
                fill_color = 'black' if self.grid_data and self.grid_data[row][col] == 1 else 'white'
                rect = canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline='gray')
                self.cells[idx][(row, col)] = rect
        # Vẽ start/end
        s, e = self.start_end
        if s in self.cells[idx]:
            x = pad_x + s[1] * cell
            y = pad_y + s[0] * cell
            canvas.create_line(x + cell*0.2, y + cell*0.2, x + cell*0.2, y + cell*0.8, fill='black', width=2)
            points = [x + cell*0.2, y + cell*0.2, x + cell*0.7, y + cell*0.4, x + cell*0.2, y + cell*0.6]
            canvas.create_polygon(points, fill='green', outline='black')
        if e in self.cells[idx]:
            x = pad_x + e[1] * cell
            y = pad_y + e[0] * cell
            center_x = x + cell/2
            center_y = y + cell/2
            radius = cell * 0.4
            for i in range(3):
                r = radius * (1 - i * 0.2)
                if r > 1:
                    canvas.create_oval(center_x-r, center_y-r, center_x+r, center_y+r, outline='red', width=2)
            dot_radius = max(2, cell * 0.1)
            canvas.create_oval(center_x-dot_radius, center_y-dot_radius, center_x+dot_radius, center_y+dot_radius, fill='red', outline='red')

    def start_compare(self):
        if self.running:
            return
        self.running = True
        self.metrics = [{}, {}]
        for i in range(2):
            for (r, c), rect in self.cells[i].items():
                color = 'black' if self.grid_data and self.grid_data[r][c] == 1 else 'white'
                (self.canvas1 if i==0 else self.canvas2).itemconfig(self.cells[i][(r, c)], fill=color)
        self.generators = [self.get_generator(self.combo1.get(), 0), self.get_generator(self.combo2.get(), 1)]
        self.metrics = [
            {'visited': 0, 'path': 0, 'start_time': time.time()},
            {'visited': 0, 'path': 0, 'start_time': time.time()}
        ]
        self.after_id_0 = self.after(self.delay, self.step_compare_0)
        self.after_id_1 = self.after(self.delay, self.step_compare_1)

    def get_generator(self, algo, idx):
        grid = self.grid_data
        s, e = self.start_end
        if algo == 'BFS':
            return self.bfs_generator(grid, s, e, idx)
        elif algo == 'DFS':
            return self.dfs_generator(grid, s, e, idx)
        elif algo == 'Dijkstra':
            return self.dijkstra_generator(grid, s, e, idx)
        else:
            return self.astar_generator(grid, s, e, idx)

    def bfs_generator(self, grid, start, goal, idx):
        from collections import deque
        visited = {start}
        queue = deque([start])
        parent = {start: None}
        while queue:
            u = queue.popleft()
            yield 'visit', u
            if u == goal:
                break
            for v in logic.get_neighbors(u, len(grid)):
                if v not in visited and grid[v[0]][v[1]] == 0:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
                    yield 'visit', v
        node = goal
        path = []
        while node is not None:
            path.append(node)
            node = parent.get(node)
        for cell in reversed(path):
            yield 'path', cell

    def dfs_generator(self, grid, start, goal, idx):
        visited = set()
        stack = [start]
        parent = {start: None}
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            yield 'visit', u
            if u == goal:
                break
            for v in logic.get_neighbors(u, len(grid)):
                if v not in visited and grid[v[0]][v[1]] == 0:
                    parent[v] = u
                    stack.append(v)
                    yield 'visit', v
        node = goal
        path = []
        while node is not None:
            path.append(node)
            node = parent.get(node)
        for cell in reversed(path):
            yield 'path', cell

    def dijkstra_generator(self, grid, start, goal, idx):
        import heapq
        dist = {start: 0}
        parent = {}
        visited = set()
        heap = [(0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            yield 'visit', u
            if u == goal:
                break
            for v, cost in logic.get_neighbors_cost(u, len(grid)):
                if grid[v[0]][v[1]] == 1:
                    continue
                nd = d + cost
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    parent[v] = u
                    heapq.heappush(heap, (nd, v))
        # Nếu không tìm được đường đi thì return luôn, tránh KeyError
        if goal not in parent and start != goal:
            return
        node = goal
        path = []
        while node != start:
            path.append(node)
            node = parent[node]
        path.append(start)
        for cell in reversed(path):
            yield 'path', cell

    def astar_generator(self, grid, start, goal, idx):
        import heapq
        g_score = {start: 0}
        f_score = {start: logic.get_heuristic(start, goal, 'Manhattan')}
        open_set = [(f_score[start], start)]
        came_from = {}
        closed = set()
        while open_set:
            _, u = heapq.heappop(open_set)
            if u in closed:
                continue
            closed.add(u)
            yield 'visit', u
            if u == goal:
                break
            for v, cost in logic.get_neighbors_cost(u, len(grid)):
                if grid[v[0]][v[1]] == 1:
                    continue
                tentative_g = g_score[u] + cost
                if v in g_score and tentative_g >= g_score[v]:
                    continue
                came_from[v] = u
                g_score[v] = tentative_g
                f_score[v] = tentative_g + logic.get_heuristic(v, goal, 'Manhattan')
                heapq.heappush(open_set, (f_score[v], v))
        node = goal
        path = []
        while node != start:
            path.append(node)
            node = came_from[node]
        path.append(start)
        for cell in reversed(path):
            yield 'path', cell

    def step_compare_0(self):
        i = 0
        if self.generators[i] is None:
            return
        try:
            typ, cell = next(self.generators[i])
            if typ == 'visit':
                self.metrics[i]['visited'] += 1
                if cell != self.start and cell != self.end:
                    color = COLORS[self.combo1.get()]['visit']
                    self.canvas1.itemconfig(self.cells[i][cell], fill=color)
            else:
                self.metrics[i]['path'] += 1
                if cell != self.start and cell != self.end:
                    color = COLORS[self.combo1.get()]['path']
                    self.canvas1.itemconfig(self.cells[i][cell], fill=color)
            elapsed = time.time() - self.metrics[i]['start_time']
            self.metric_labels[2][i].configure(text=f'{elapsed:.2f}')
            self.metric_labels[0][i].configure(text=str(self.metrics[i]['visited']))
            self.metric_labels[1][i].configure(text=str(self.metrics[i]['path']))
            self.after_id_0 = self.after(self.delay, self.step_compare_0)
        except StopIteration:
            elapsed = time.time() - self.metrics[i]['start_time']
            self.metric_labels[2][i].configure(text=f'{elapsed:.2f}')
            self.metric_labels[0][i].configure(text=str(self.metrics[i]['visited']))
            self.metric_labels[1][i].configure(text=str(self.metrics[i]['path']))
            # Nếu cả hai đều xong thì cho phép chạy lại
            if (i == 0 and (self.generators[1] is None or not hasattr(self.generators[1], '__next__'))) or (i == 1 and (self.generators[0] is None or not hasattr(self.generators[0], '__next__'))):
                self.running = False
            return

    def step_compare_1(self):
        i = 1
        if self.generators[i] is None:
            return
        try:
            typ, cell = next(self.generators[i])
            if typ == 'visit':
                self.metrics[i]['visited'] += 1
                if cell != self.start and cell != self.end:
                    color = COLORS[self.combo2.get()]['visit']
                    self.canvas2.itemconfig(self.cells[i][cell], fill=color)
            else:
                self.metrics[i]['path'] += 1
                if cell != self.start and cell != self.end:
                    color = COLORS[self.combo2.get()]['path']
                    self.canvas2.itemconfig(self.cells[i][cell], fill=color)
            elapsed = time.time() - self.metrics[i]['start_time']
            self.metric_labels[2][i].configure(text=f'{elapsed:.2f}')
            self.metric_labels[0][i].configure(text=str(self.metrics[i]['visited']))
            self.metric_labels[1][i].configure(text=str(self.metrics[i]['path']))
            self.after_id_1 = self.after(self.delay, self.step_compare_1)
        except StopIteration:
            elapsed = time.time() - self.metrics[i]['start_time']
            self.metric_labels[2][i].configure(text=f'{elapsed:.2f}')
            self.metric_labels[0][i].configure(text=str(self.metrics[i]['visited']))
            self.metric_labels[1][i].configure(text=str(self.metrics[i]['path']))
            # Nếu cả hai đều xong thì cho phép chạy lại
            if (i == 0 and (self.generators[1] is None or not hasattr(self.generators[1], '__next__'))) or (i == 1 and (self.generators[0] is None or not hasattr(self.generators[0], '__next__'))):
                self.running = False
            return

if __name__ == '__main__':
    app = MazeCompareApp()
    app.mainloop() 
