
import customtkinter as ctk
import tkinter as tk
import WOM_MAZE_LOGIC as logic
import time
import heapq
import random
import threading

# Configure CustomTkinter theme
ctk.set_appearance_mode('light')
ctk.set_default_color_theme('green')

class MudMazeApp(ctk.CTk):
    """Cửa sổ giải bài toán Mud Maze."""
    def __init__(self):
        super().__init__()
        self.title('Mud Maze')
        self.state('zoomed')
        self.grid_size = 20  # Default grid size
        self.max_grid_size = 100  # Maximum grid size
        self.grid_data = None
        self.terrain_data = None  # Store terrain information
        self.start = None
        self.end = None
        self.cell_size = None
        self.pad_x = None
        self.pad_y = None
        self.speed = 5  # Default animation speed
        self.delay = 200  # Animation delay in ms

        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header = ctk.CTkFrame(main_container, fg_color='#8B4513')  # Brown color
        header.pack(fill=tk.X, pady=(0, 10))
        ctk.CTkLabel(header, text='Mud Maze', 
                    font=('Arial', 24, 'bold'), 
                    text_color='white').pack(pady=10)

        # Content area
        content = ctk.CTkFrame(main_container)
        content.pack(fill=tk.BOTH, expand=True)

        # Left control panel
        control = ctk.CTkFrame(content, width=300, fg_color='#8B4513')  # Brown color
        control.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control.pack_propagate(False)

        # Control panel title
        ctk.CTkLabel(control, text='Control Panel', 
                    font=('Arial', 18, 'bold'),
                    text_color='white').pack(pady=(20, 10))

        # Grid size control (now from 20 to 100)
        ctk.CTkLabel(control, text='Grid Size', 
                    text_color='white').pack(pady=(10, 2))
        self.size_slider = ctk.CTkSlider(control, from_=20, to=100, 
                                       command=self.update_grid_size)
        self.size_slider.set(self.grid_size)
        self.size_slider.pack(pady=(0, 5))
        self.size_label = ctk.CTkLabel(control, text=str(self.grid_size),
                                     text_color='white')
        self.size_label.pack()

        # Maze generation options
        ctk.CTkLabel(control, text='Maze Generation', 
                    text_color='white').pack(pady=(20, 2))
        self.maze_type = ctk.CTkComboBox(control, 
                                       values=['Recursive Backtracking', 'Prim', 'Kruskal'],
                                       width=200)
        self.maze_type.set('Recursive Backtracking')
        self.maze_type.pack(pady=(0, 10))

        # Generate button
        self.generate_btn = ctk.CTkButton(control, text='Generate Maze',
                                        command=self.generate_maze,
                                        width=200)
        self.generate_btn.pack(pady=5)

        # Animation speed control
        ctk.CTkLabel(control, text='Animation Speed', 
                    text_color='white').pack(pady=(20, 2))
        self.speed_slider = ctk.CTkSlider(control, from_=1, to=10,
                                        command=self.update_speed)
        self.speed_slider.set(self.speed)
        self.speed_slider.pack(pady=(0, 5))
        self.speed_label = ctk.CTkLabel(control, text=str(self.speed),
                                      text_color='white')
        self.speed_label.pack()

        # Find Path button
        self.find_path_btn = ctk.CTkButton(control, text='Find Path',
                                         command=self.start_pathfinding,
                                         width=200)
        self.find_path_btn.pack(pady=5)

        # Reset Maze button
        self.reset_btn = ctk.CTkButton(control, text='Reset Maze',
                                       command=self.reset_maze,
                                       width=200)
        self.reset_btn.pack(pady=10)

        # Metrics frame
        metrics_frame = ctk.CTkFrame(control, fg_color='#8B4513')
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)

        # Visited cells count
        ctk.CTkLabel(metrics_frame, text='Cells Visited:', 
                    text_color='white').pack(pady=(5, 2))
        self.visited_label = ctk.CTkLabel(metrics_frame, text='0',
                                        text_color='white')
        self.visited_label.pack()

        # Path length
        ctk.CTkLabel(metrics_frame, text='Path Length:', 
                    text_color='white').pack(pady=(5, 2))
        self.path_length_label = ctk.CTkLabel(metrics_frame, text='0',
                                            text_color='white')
        self.path_length_label.pack()

        # Time elapsed
        ctk.CTkLabel(metrics_frame, text='Time Elapsed (s):', 
                    text_color='white').pack(pady=(5, 2))
        self.time_label = ctk.CTkLabel(metrics_frame, text='0.00',
                                     text_color='white')
        self.time_label.pack()

        # Canvas area (no right control panel)
        canvas_frame = ctk.CTkFrame(content, fg_color='white')
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Bind events
        self.canvas.bind('<Configure>', lambda e: self.draw_grid(self.grid_size))
        self.canvas.bind('<Button-1>', self.on_canvas_left_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)

        # Initialize grid
        self.draw_grid(self.grid_size)

    # ====== Tiện ích vẽ và thao tác lưới ======
    def clear_path_effects(self):
        """Xóa hiệu ứng màu của đường đi cũ và visited cũ, chỉ giữ lại tường và icon start/end."""
        for (r, c), rect in self.cells.items():
            if self.grid_data and self.grid_data[r][c] == 1:
                self.canvas.itemconfig(rect, fill='#4A2F1B')
            else:
                self.canvas.itemconfig(rect, fill='white')
        self.draw_start_end_icons()
        self.visited_count = 0
        self.path_length = 0
        self.visited_label.configure(text='0')
        self.path_length_label.configure(text='0')
        self.time_label.configure(text='0.00')

    def randomly_update_maze(self):
        """Randomly open/close some walls in the maze, except start/end, and always keep a path from start to end."""
        if not self.grid_data:
            return
        n = self.grid_size
        old_grid = [row[:] for row in self.grid_data]
        num_changes = max(1, n // 5)
        for _ in range(num_changes):
            r = random.randint(0, n-1)
            c = random.randint(0, n-1)
            if (r, c) == self.start or (r, c) == self.end:
                continue
            self.grid_data[r][c] = 0 if self.grid_data[r][c] == 1 else 1
        if not self._has_path():
            self.grid_data = old_grid
        self.draw_grid(n)

    def _has_path(self):
        """Kiểm tra còn đường đi từ start đến end không (BFS)."""
        from collections import deque
        n = self.grid_size
        visited = set()
        queue = deque([self.start])
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.end:
                return True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                    if self.grid_data[nr][nc] == 0:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        return False

    def draw_grid(self, n):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w < 10 or h < 10:
            self.after(50, lambda: self.draw_grid(n))
            return
        self.canvas.delete('all')
        cell = min(w / n, h / n)
        total = cell * n
        pad_x = (w - total) / 2
        pad_y = (h - total) / 2
        self.cell_size = cell
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.cells = {}
        for row in range(n):
            for col in range(n):
                x1 = pad_x + col * cell
                y1 = pad_y + row * cell
                x2 = x1 + cell
                y2 = y1 + cell
                if self.grid_data is None:
                    fill_color = 'white'
                elif self.grid_data[row][col] == 1:
                    fill_color = '#4A2F1B'  # Wall
                else:
                    fill_color = 'white'    # Path
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline='#E0E0E0', width=1)
                self.cells[(row, col)] = rect
        if self.start not in self.cells:
            self.start = (0, 0)
        if self.end not in self.cells:
            self.end = (n-1, n-1)
        self.draw_start_end_icons()
        self.canvas.update_idletasks()
        self.canvas.update()

    def draw_start_end_icons(self):
        """Draw start and end icons on the grid."""
        if not hasattr(self, 'cells'):
            return
        for item in self.canvas.find_all():
            if item not in self.cells.values():
                self.canvas.delete(item)
        # Start icon
        if self.start in self.cells:
            x = self.pad_x + self.start[1] * self.cell_size
            y = self.pad_y + self.start[0] * self.cell_size
            self.canvas.create_line(x + self.cell_size*0.2, y + self.cell_size*0.2,
                                  x + self.cell_size*0.2, y + self.cell_size*0.8,
                                  fill='#8B4513', width=2)
            points = [
                x + self.cell_size*0.2, y + self.cell_size*0.2,
                x + self.cell_size*0.7, y + self.cell_size*0.4,
                x + self.cell_size*0.2, y + self.cell_size*0.6
            ]
            self.canvas.create_polygon(points, fill='#8B4513', outline='#8B4513')
        # End icon
        if self.end in self.cells:
            x = self.pad_x + self.end[1] * self.cell_size
            y = self.pad_y + self.end[0] * self.cell_size
            center_x = x + self.cell_size/2
            center_y = y + self.cell_size/2
            max_radius = int(self.cell_size * 0.4)
            step = max(1, int(self.cell_size * 0.1))
            for r in range(max_radius, 0, -step):
                self.canvas.create_oval(center_x-r, center_y-r,
                                     center_x+r, center_y+r,
                                     outline='#8B4513', width=2)
            dot_radius = max(1, int(self.cell_size * 0.1))
            self.canvas.create_oval(center_x-dot_radius,
                                  center_y-dot_radius,
                                  center_x+dot_radius,
                                  center_y+dot_radius,
                                  fill='#8B4513', outline='#8B4513')

    def pixel_to_cell(self, x, y):
        """Convert canvas coordinates to grid cell indices."""
        if self.cell_size is None:
            return None, None
        col = int((x - self.pad_x) / self.cell_size)
        row = int((y - self.pad_y) / self.cell_size)
        return row, col

    # ====== Xử lý sự kiện và logic chính ======
    def on_canvas_left_click(self, event):
        """Handle left-click to set start point."""
        r, c = self.pixel_to_cell(event.x, event.y)
        if r is None or not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return
        if self.grid_data is not None and self.grid_data[r][c] == 1:
            return  # Do not allow setting start on a wall
        if self.end == (r, c):
            return
        # Xóa hiệu ứng cũ
        self.clear_path_effects()
        # Set new start point
        self.start = (r, c)
        self.draw_start_end_icons()
        # Cho phép mê cung động lại nếu cần
        self._last_maze_update = time.time()
        if hasattr(self, '_avoid'):
            del self._avoid

    def on_canvas_right_click(self, event):
        """Handle right-click to set end point."""
        r, c = self.pixel_to_cell(event.x, event.y)
        if r is None or not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return
        if self.grid_data is not None and self.grid_data[r][c] == 1:
            return  # Do not allow setting end on a wall
        if self.start == (r, c):
            return
        # Xóa hiệu ứng cũ
        self.clear_path_effects()
        # Set new end point
        self.end = (r, c)
        self.draw_start_end_icons()
        # Cho phép mê cung động lại nếu cần
        self._last_maze_update = time.time()
        if hasattr(self, '_avoid'):
            del self._avoid

    def update_grid_size(self, value):
        """Update grid size from slider."""
        new_size = int(value)
        if new_size > self.max_grid_size:
            new_size = self.max_grid_size
            self.size_slider.set(new_size)
        self.grid_size = new_size
        self.size_label.configure(text=str(new_size))
        self.draw_grid(new_size)

    def update_speed(self, value):
        """Update animation speed from slider."""
        self.speed = int(value)
        self.speed_label.configure(text=str(self.speed))
        self.delay = int(200 / self.speed)  # Lower delay = faster speed

    def generate_maze(self):
        """Generate a new maze (walls and paths only, no special terrain)."""
        n = self.grid_size
        if n > self.max_grid_size:
            n = self.max_grid_size
            self.grid_size = n
            self.size_slider.set(n)
            self.size_label.configure(text=str(n))
        self.grid_data = logic.generate_maze(n, self.maze_type.get(), 'Mud Maze')
        self.terrain_data = None  # No special terrain
        # Add loops for alternative paths
        if self.start is not None and self.end is not None:
            loops = min(3, n // 20)
            self.grid_data = logic.add_targeted_loops(self.grid_data, self.start, self.end, loops=loops)
        # Ensure start/end are not on a wall
        self.start = self.find_nearest_empty((0, 0))
        self.end = self.find_nearest_empty((n-1, n-1))
        # Draw the maze instantly (no animation)
        self.draw_grid(n)
        # Reset metrics
        self.visited_count = 0
        self.path_length = 0
        self.visited_label.configure(text='0')
        self.path_length_label.configure(text='0')
        self.time_label.configure(text='0.00')
        self.draw_start_end_icons()
        self.canvas.update_idletasks()
        self.canvas.update()

    def reset_maze(self):
        """Reset the maze to its initial state."""
        self.grid_data = None
        n = self.grid_size
        start = self.find_nearest_empty((0, 0))
        end = self.find_nearest_empty((n-1, n-1))
        # Ensure start and end are not the same
        if start == end:
            # Try to find another empty cell for end, searching from bottom right
            for r in range(n-1, -1, -1):
                for c in range(n-1, -1, -1):
                    if (r, c) != start:
                        if self.grid_data is None or (self.grid_data[r][c] == 0):
                            end = (r, c)
                            break
                if end != start:
                    break
        self.start = start
        self.end = end
        self.visited_count = 0
        self.path_length = 0
        self.visited_label.configure(text='0')
        self.path_length_label.configure(text='0')
        self.time_label.configure(text='0.00')
        self.draw_grid(n)
        self.canvas.update_idletasks()
        self.canvas.update()

    def start_pathfinding(self):
        """Start the A* pathfinding animation (always reset effects and generator)."""
        if not self.start or not self.end:
            return
        # Xóa hiệu ứng cũ trước khi tìm đường mới
        self.clear_path_effects()
        # Reset metrics
        self.visited_count = 0
        self.path_length = 0
        self.start_time = time.time()
        # Reset grid colors nhưng giữ tường
        for (r, c), rect in self.cells.items():
            if self.grid_data and self.grid_data[r][c] == 1:
                self.canvas.itemconfig(rect, fill='#4A2F1B')
            else:
                self.canvas.itemconfig(rect, fill='white')
        # Redraw start/end icons
        self.draw_start_end_icons()
        # Reset _avoid để replan đúng
        if hasattr(self, '_avoid'):
            del self._avoid
        # Nếu chưa có mê cung, tạo grid toàn đường đi
        if self.grid_data is None:
            grid = [[0] * self.grid_size for _ in range(self.grid_size)]
        else:
            grid = self.grid_data
        self.step_gen = astar_generator(grid, self.start, self.end)
        self._last_maze_update = time.time()
        self.after(self.delay, self.step)

    def step(self):
        """Step animation with replan if path is blocked, maze changes only during visit phase."""
        try:
            typ, cell = next(self.step_gen)
            is_wall = self.grid_data is not None and self.grid_data[cell[0]][cell[1]] == 1
            if typ == 'visit':
                self.visited_count += 1
                if cell != self.start and cell != self.end:
                    if not is_wall:
                        self.canvas.itemconfig(self.cells[cell], fill='#ADD8E6')
                self.visited_label.configure(text=str(self.visited_count))
                self.time_label.configure(text=f"{time.time() - self.start_time:.2f}")
                # Chỉ thay đổi mê cung trong giai đoạn visit
                now = time.time()
                if not hasattr(self, '_last_maze_update'):
                    self._last_maze_update = now
                if now - self._last_maze_update >= 1.0:
                    self.randomly_update_maze()
                    self._last_maze_update = now
            else:  # 'path'
                # Khi bắt đầu vẽ đường đi ngắn nhất thì dừng thay đổi mê cung
                self._last_maze_update = float('inf')
                if is_wall:
                    if not hasattr(self, '_avoid'): self._avoid = set()
                    self._avoid.add(cell)
                    self.step_gen = astar_generator(self.grid_data if self.grid_data is not None else [[0]*self.grid_size for _ in range(self.grid_size)], cell, self.end, avoid=self._avoid)
                    return self.after(self.delay, self.step)
                self.path_length += 1
                if cell != self.start and cell != self.end:
                    if not is_wall:
                        self.canvas.itemconfig(self.cells[cell], fill='blue')
                self.path_length_label.configure(text=str(self.path_length))
                self.time_label.configure(text=f"{time.time() - self.start_time:.2f}")
            self.after(self.delay, self.step)
        except StopIteration:
            self._last_maze_update = float('inf')
            elapsed = time.time() - self.start_time
            self.visited_label.configure(text=str(self.visited_count))
            self.path_length_label.configure(text=str(self.path_length))
            self.time_label.configure(text=f"{elapsed:.2f}")

    def find_nearest_empty(self, pos):
        """Tìm ô trống (không phải tường) gần nhất từ vị trí pos."""
        from collections import deque
        n = self.grid_size
        visited = set()
        queue = deque([pos])
        while queue:
            r, c = queue.popleft()
            if self.grid_data is None or self.grid_data[r][c] == 0:
                return (r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return pos

def astar_generator(grid, start, goal, heuristic='Manhattan', avoid=None):
    import heapq
    from WOM_MAZE_LOGIC import get_neighbors_cost, get_heuristic
    n = len(grid)
    if avoid is None:
        avoid = set()
    g_score = {start: 0}
    f_score = {start: get_heuristic(start, goal, heuristic)}
    open_set = [(f_score[start], start)]
    came_from = {}
    closed = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if grid[current[0]][current[1]] == 1 or current in avoid:
            continue
        if current == goal:
            break
        if current in closed:
            continue
        closed.add(current)
        yield 'visit', current
        for v, cost in get_neighbors_cost(current, n):
            if grid[v[0]][v[1]] == 1 or v in avoid:
                continue
            tentative_g = g_score[current] + cost
            if v in g_score and tentative_g >= g_score[v]:
                continue
            came_from[v] = current
            g_score[v] = tentative_g
            f_score[v] = tentative_g + get_heuristic(v, goal, heuristic)
            heapq.heappush(open_set, (f_score[v], v))
    if goal not in came_from and start != goal:
        return
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = came_from.get(node, start)
        if node == start:
            break
    path.append(start)
    for cell in reversed(path):
        yield 'path', cell

if __name__ == '__main__':
    app = MudMazeApp()
    app.mainloop() 
