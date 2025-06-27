
import customtkinter as ctk
import tkinter as tk
import WOM_MAZE_LOGIC as logic
import time
import heapq
import random

# Configure CustomTkinter theme
ctk.set_appearance_mode('light')
ctk.set_default_color_theme('green')

# Constants for terrain types
TERRAIN_TYPES = {
    'DEFAULT': {'color': 'white', 'fuel_cost': 1, 'name': 'Default'},
    'GRASS': {'color': '#90EE90', 'fuel_cost': 2, 'name': 'Grass'},
    'MUD': {'color': '#8B4513', 'fuel_cost': 3, 'name': 'Mud'},
    'SAND': {'color': '#F4A460', 'fuel_cost': 4, 'name': 'Sand'},
    'ROCK': {'color': '#808080', 'fuel_cost': 5, 'name': 'Rock'}
}

# --- Khởi tạo & UI ---
class EcoBotNavigatorApp(ctk.CTk):
    """Cửa sổ giải bài toán EcoBot Navigator."""
    def __init__(self):
        super().__init__()
        self.title('EcoBot Navigator')
        self.state('zoomed')
        self.grid_size = 20  # Default grid size
        self.grid_data = None
        self.terrain_data = None  # New: Store terrain information
        self.fuel_stations = []   # New: Store fuel station positions
        self.start = None
        self.end = None
        self.cell_size = None
        self.pad_x = None
        self.pad_y = None
        self.speed = 5  # Default animation speed
        self.delay = 200  # Animation delay in ms
        self.current_fuel = 20  # Initial fuel
        self.max_fuel = 20      # Maximum fuel capacity

        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        header = ctk.CTkFrame(main_container, fg_color='#00a000')
        header.pack(fill=tk.X, pady=(0, 10))
        ctk.CTkLabel(header, text='EcoBot Navigator', 
                    font=('Arial', 24, 'bold'), 
                    text_color='white').pack(pady=10)

        # Content area
        content = ctk.CTkFrame(main_container)
        content.pack(fill=tk.BOTH, expand=True)

        # Left control panel
        control = ctk.CTkFrame(content, width=300, fg_color='#00a000')
        control.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control.pack_propagate(False)

        # Control panel title
        ctk.CTkLabel(control, text='Control Panel', 
                    font=('Arial', 18, 'bold'),
                    text_color='white').pack(pady=(20, 10))

        # Grid size control
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

        # Terrain density slider
        ctk.CTkLabel(control, text='Terrain Density (%)', 
                    text_color='white').pack(pady=(20, 2))
        self.terrain_slider = ctk.CTkSlider(control, from_=0, to=50,
                                          command=self.update_terrain_density)
        self.terrain_slider.set(30)  # Default 30%
        self.terrain_slider.pack(pady=(0, 5))
        self.terrain_label = ctk.CTkLabel(control, text='30%',
                                        text_color='white')
        self.terrain_label.pack()

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

        # Metrics frame
        metrics_frame = ctk.CTkFrame(control, fg_color='#008000')
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

        # Add fuel information to metrics frame
        ctk.CTkLabel(metrics_frame, text='Current Fuel:', 
                    text_color='white').pack(pady=(5, 2))
        self.fuel_label = ctk.CTkLabel(metrics_frame, text='20',
                                     text_color='white')
        self.fuel_label.pack()
        # Add remaining fuel label
        self.remaining_fuel_label = ctk.CTkLabel(metrics_frame, text='Remaining Fuel: 20', text_color='white')
        self.remaining_fuel_label.pack()

        # Add final fuel display
        ctk.CTkLabel(metrics_frame, text='Final Fuel:', 
                    text_color='white').pack(pady=(5, 2))
        self.final_fuel_label = ctk.CTkLabel(metrics_frame, text='20',
                                           text_color='white')
        self.final_fuel_label.pack()

        # Add fuel information panel
        fuel_info_frame = ctk.CTkFrame(control, fg_color='#008000')
        fuel_info_frame.pack(fill=tk.X, padx=10, pady=10)

        ctk.CTkLabel(fuel_info_frame, text='Fuel Information', 
                    font=('Arial', 14, 'bold'),
                    text_color='white').pack(pady=(5, 5))

        # Fuel status
        self.fuel_status_label = ctk.CTkLabel(fuel_info_frame, 
                                            text='Status: Ready',
                                            text_color='white')
        self.fuel_status_label.pack(pady=2)

        # Fuel consumption
        self.fuel_consumption_label = ctk.CTkLabel(fuel_info_frame, 
                                                 text='Last Move: 0 fuel',
                                                 text_color='white')
        self.fuel_consumption_label.pack(pady=2)

        # Fuel stations visited
        self.fuel_stations_visited_label = ctk.CTkLabel(fuel_info_frame, 
                                                      text='Stations Visited: 0',
                                                      text_color='white')
        self.fuel_stations_visited_label.pack(pady=2)

        # Canvas area
        canvas_frame = ctk.CTkFrame(content, fg_color='white')
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Right control panel
        right_control = ctk.CTkFrame(content, width=300, fg_color='#00a000')
        right_control.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0))
        right_control.pack_propagate(False)

        # Right panel title
        ctk.CTkLabel(right_control, text='Information Panel', 
                    font=('Arial', 18, 'bold'),
                    text_color='white').pack(pady=(20, 10))

        # Terrain information frame
        terrain_frame = ctk.CTkFrame(right_control, fg_color='#008000')
        terrain_frame.pack(fill=tk.X, padx=10, pady=5)

        ctk.CTkLabel(terrain_frame, text='Terrain Types & Fuel Cost', 
                    font=('Arial', 14, 'bold'),
                    text_color='white').pack(pady=(5, 5))

        # Create terrain type indicators with fuel costs
        for terrain, info in TERRAIN_TYPES.items():
            terrain_item = ctk.CTkFrame(terrain_frame, fg_color='transparent')
            terrain_item.pack(fill=tk.X, padx=5, pady=2)
            
            # Color indicator
            color_indicator = ctk.CTkFrame(terrain_item, width=20, height=20, 
                                         fg_color=info['color'])
            color_indicator.pack(side=tk.LEFT, padx=5)
            
            # Terrain name and fuel cost
            ctk.CTkLabel(terrain_item, 
                        text=f"{terrain}: {info['fuel_cost']} fuel",
                        text_color='white').pack(side=tk.LEFT, padx=5)

        # Fuel station information
        fuel_frame = ctk.CTkFrame(right_control, fg_color='#008000')
        fuel_frame.pack(fill=tk.X, padx=10, pady=5)

        ctk.CTkLabel(fuel_frame, text='Fuel Stations', 
                    font=('Arial', 14, 'bold'),
                    text_color='white').pack(pady=(5, 5))

        ctk.CTkLabel(fuel_frame, 
                    text='Each station provides 20 fuel',
                    text_color='white').pack(pady=5)

        self.fuel_count_label = ctk.CTkLabel(fuel_frame, 
                                           text='Number of stations: 0',
                                           text_color='white')
        self.fuel_count_label.pack(pady=5)

        # Path information frame
        path_frame = ctk.CTkFrame(right_control, fg_color='#008000')
        path_frame.pack(fill=tk.X, padx=10, pady=5)

        ctk.CTkLabel(path_frame, text='Path Information', 
                    font=('Arial', 14, 'bold'),
                    text_color='white').pack(pady=(5, 5))

        # Total path cost
        self.total_cost_label = ctk.CTkLabel(path_frame, 
                                           text='Total Path Cost: 0.0',
                                           text_color='white')
        self.total_cost_label.pack(pady=5)

        # Terrain distribution
        self.terrain_dist_label = ctk.CTkLabel(path_frame, 
                                             text='Terrain Distribution:',
                                             text_color='white')
        self.terrain_dist_label.pack(pady=5)

        # Reset button
        self.reset_btn = ctk.CTkButton(right_control, 
                                     text='Reset Maze',
                                     command=self.on_reset,
                                     width=200)
        self.reset_btn.pack(pady=20)

        # Bind events
        self.canvas.bind('<Configure>', lambda e: self.draw_grid(self.grid_size))
        self.canvas.bind('<Button-1>', self.on_canvas_left_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)

        # Initialize grid
        self.draw_grid(self.grid_size)

    # ===== Tiện ích và cập nhật giao diện =====
    def blend_colors(self, color1, color2, alpha=0.5):
        def color_to_hex(color):
            if color.startswith('#'):
                return color
            color_map = {
                'white': '#FFFFFF', 'black': '#000000', 'blue': '#0000FF',
                'red': '#FF0000', 'green': '#00FF00', 'yellow': '#FFFF00',
                'orange': '#FFA500', 'purple': '#800080', 'gray': '#808080', 'lightblue': '#ADD8E6'
            }
            return color_map.get(color.lower(), '#FFFFFF')
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        hex1 = color_to_hex(color1)
        hex2 = color_to_hex(color2)
        rgb1 = hex_to_rgb(hex1)
        rgb2 = hex_to_rgb(hex2)
        blended = tuple(int(rgb1[i] * (1 - alpha) + rgb2[i] * alpha) for i in range(3))
        return rgb_to_hex(blended)

    def pixel_to_cell(self, x, y):
        if self.cell_size is None:
            return None, None
        col = int((x - self.pad_x) / self.cell_size)
        row = int((y - self.pad_y) / self.cell_size)
        return row, col

    def draw_start_end_icons(self):
        if not hasattr(self, 'cells'):
            return
        for item in self.canvas.find_all():
            if item not in self.cells.values():
                self.canvas.delete(item)
        if self.start in self.cells:
            x = self.pad_x + self.start[1] * self.cell_size
            y = self.pad_y + self.start[0] * self.cell_size
            self.canvas.create_line(x + self.cell_size*0.2, y + self.cell_size*0.2,
                                  x + self.cell_size*0.2, y + self.cell_size*0.8,
                                  fill='black', width=2)
            points = [
                x + self.cell_size*0.2, y + self.cell_size*0.2,
                x + self.cell_size*0.7, y + self.cell_size*0.4,
                x + self.cell_size*0.2, y + self.cell_size*0.6
            ]
            self.canvas.create_polygon(points, fill='green', outline='black')
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
                                     outline='red', width=2)
            dot_radius = max(1, int(self.cell_size * 0.1))
            self.canvas.create_oval(center_x-dot_radius,
                                  center_y-dot_radius,
                                  center_x+dot_radius,
                                  center_y+dot_radius,
                                  fill='red', outline='red')

    def draw_fuel_stations(self):
        if not hasattr(self, 'cells'):
            return
        for item in self.canvas.find_all():
            if 'fuel_station' in self.canvas.gettags(item):
                self.canvas.delete(item)
        for station in self.fuel_stations:
            x = self.pad_x + station[1] * self.cell_size
            y = self.pad_y + station[0] * self.cell_size
            self.canvas.create_rectangle(x + self.cell_size*0.3, y + self.cell_size*0.4,
                                      x + self.cell_size*0.7, y + self.cell_size*0.8,
                                      fill='#FFD700', outline='black', tags='fuel_station')
            self.canvas.create_line(x + self.cell_size*0.5, y + self.cell_size*0.3,
                                  x + self.cell_size*0.5, y + self.cell_size*0.4,
                                  fill='black', width=2, tags='fuel_station')
            self.canvas.create_text(x + self.cell_size*0.5, y + self.cell_size*0.6,
                                  text='⛽', font=('Arial', int(self.cell_size*0.4)), tags='fuel_station')

    def update_information_panel(self):
        self.fuel_count_label.configure(text=f'Number of stations: {len(self.fuel_stations)}')
        if self.terrain_data:
            terrain_counts = {terrain: 0 for terrain in TERRAIN_TYPES}
            for row in self.terrain_data:
                for cell in row:
                    if cell:
                        terrain_counts[list(TERRAIN_TYPES.keys())[list(TERRAIN_TYPES.values()).index(cell)]] += 1
            dist_text = 'Terrain Distribution:\n'
            for terrain, count in terrain_counts.items():
                if count > 0:
                    dist_text += f'{terrain}: {count} cells\n'
            self.terrain_dist_label.configure(text=dist_text)
        else:
            self.terrain_dist_label.configure(text='Terrain Distribution: None')
        self.total_cost_label.configure(text='Total Path Cost: 0.0')

    def update_fuel_display(self):
        self.fuel_label.configure(text=str(self.current_fuel))
        if self.current_fuel <= 5:
            status = "Low Fuel!"; color = "red"
        elif self.current_fuel <= 10:
            status = "Warning"; color = "orange"
        else:
            status = "Good"; color = "white"
        self.fuel_status_label.configure(text=f'Status: {status}', text_color=color)

    def update_grid_size(self, value):
        self.grid_size = int(value)
        self.size_label.configure(text=str(self.grid_size))
        self.draw_grid(self.grid_size)

    def update_speed(self, value):
        self.speed = int(value)
        self.speed_label.configure(text=str(self.speed))
        self.delay = int(200 / self.speed)

    def update_terrain_density(self, value):
        self.terrain_label.configure(text=f'{int(value)}%')

    def draw_grid(self, n):
        self.canvas.delete('all')
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
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
                if self.grid_data and self.grid_data[row][col] == 1:
                    fill_color = 'black'
                elif self.terrain_data and self.terrain_data[row][col]:
                    fill_color = self.terrain_data[row][col]['color']
                else:
                    fill_color = 'white'
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline='gray')
                self.cells[(row, col)] = rect
        if self.start not in self.cells:
            self.start = (0, 0)
        if self.end not in self.cells:
            self.end = (n-1, n-1)
        self.draw_start_end_icons()
        self.draw_fuel_stations()

    # ===== Xử lý sự kiện và logic chính =====
    def on_canvas_left_click(self, event):
        r, c = self.pixel_to_cell(event.x, event.y)
        if r is None or not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return
        if self.grid_data is not None and self.grid_data[r][c] == 1:
            return
        if self.end == (r, c):
            return
        old_terrain = None
        if self.start is not None and self.terrain_data is not None:
            old_terrain = self.terrain_data[self.start[0]][self.start[1]]
        if self.start is not None and self.start != self.end:
            if self.grid_data is not None:
                if self.grid_data[self.start[0]][self.start[1]] == 1:
                    color = 'black'
                elif old_terrain:
                    color = old_terrain['color']
                else:
                    color = 'white'
                self.canvas.itemconfig(self.cells[self.start], fill=color)
        self.start = (r, c)
        self.draw_start_end_icons()
        self.draw_fuel_stations()

    def on_canvas_right_click(self, event):
        r, c = self.pixel_to_cell(event.x, event.y)
        if r is None or not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return
        if self.grid_data is not None and self.grid_data[r][c] == 1:
            return
        if self.start == (r, c):
            return
        old_terrain = None
        if self.end is not None and self.terrain_data is not None:
            old_terrain = self.terrain_data[self.end[0]][self.end[1]]
        if self.end is not None and self.end != self.start:
            if self.grid_data is not None:
                if self.grid_data[self.end[0]][self.end[1]] == 1:
                    color = 'black'
                elif old_terrain:
                    color = old_terrain['color']
                else:
                    color = 'white'
                self.canvas.itemconfig(self.cells[self.end], fill=color)
        self.end = (r, c)
        self.draw_start_end_icons()
        self.draw_fuel_stations()

    def on_reset(self):
        self.grid_data = None
        self.terrain_data = None
        self.fuel_stations = []
        self.start = None
        self.end = None
        self.current_fuel = self.max_fuel
        self.update_fuel_display()
        self.draw_grid(self.grid_size)
        self.update_information_panel()

    def generate_maze(self):
        n = self.grid_size
        self.grid_data = logic.generate_maze(n, self.maze_type.get(), 'EcoBot Navigator')
        self.terrain_data = [[TERRAIN_TYPES['DEFAULT'] for _ in range(n)] for _ in range(n)]
        terrain_density = int(self.terrain_slider.get()) / 100.0
        path_cells = [(r, c) for r in range(n) for c in range(n) if self.grid_data[r][c] == 0]
        num_terrain_cells = int(len(path_cells) * terrain_density)
        terrain_cells = random.sample(path_cells, num_terrain_cells)
        for r, c in terrain_cells:
            terrain_type = random.choice(['GRASS', 'MUD', 'SAND', 'ROCK'])
            self.terrain_data[r][c] = TERRAIN_TYPES[terrain_type]
        num_stations = max(4, n * n // 25)
        self.fuel_stations = []
        available_cells = [cell for cell in path_cells if cell not in terrain_cells]
        if available_cells:
            self.fuel_stations = random.sample(available_cells, min(num_stations, len(available_cells)))
        self.current_fuel = self.max_fuel
        self.update_fuel_display()
        self.fuel_consumption_label.configure(text='Last Move: 0 fuel')
        self.fuel_stations_visited_label.configure(text='Stations Visited: 0')
        self.final_fuel_label.configure(text='20')
        self.draw_grid(n)
        self.visited_count = 0
        self.path_length = 0
        self.visited_label.configure(text='0')
        self.path_length_label.configure(text='0')
        self.time_label.configure(text='0.00')
        self.update_information_panel()

    def start_pathfinding(self):
        if not self.start or not self.end:
            return
        self.visited_count = 0
        self.path_length = 0
        self.start_time = time.time()
        for (r, c), rect in self.cells.items():
            if self.grid_data and self.grid_data[r][c] == 1:
                self.canvas.itemconfig(rect, fill='black')
            elif self.terrain_data and self.terrain_data[r][c]:
                self.canvas.itemconfig(rect, fill=self.terrain_data[r][c]['color'])
            else:
                self.canvas.itemconfig(rect, fill='white')
        self.draw_start_end_icons()
        self.draw_fuel_stations()
        grid = self.grid_data if self.grid_data is not None else [[0] * self.grid_size for _ in range(self.grid_size)]
        self.step_gen = self.astar_generator(grid, self.start, self.end)
        # Kiểm tra nếu không tìm được đường đi thì báo lên UI
        try:
            first = next(self.step_gen)
            self.step_gen = self._prepend_first(first, self.step_gen)
            self.after(self.delay, self.step)
        except StopIteration:
            self.time_label.configure(text='No path found!')
            self.visited_label.configure(text='0')
            self.path_length_label.configure(text='0')
            self.remaining_fuel_label.configure(text=f'Remaining Fuel: {self.current_fuel}')
            self.final_fuel_label.configure(text=str(self.current_fuel))

    def step(self):
        try:
            typ, cell = next(self.step_gen)
            if typ == 'visit':
                self.visited_count += 1
                if cell != self.start and cell != self.end:
                    if not (self.grid_data and self.grid_data[cell[0]][cell[1]] == 1):
                        self.canvas.itemconfig(self.cells[cell], fill='#ADD8E6')
                self.visited_label.configure(text=str(self.visited_count))
                self.time_label.configure(text=f"{time.time() - self.start_time:.2f}")
            else:  # 'path'
                self.path_length += 1
                if cell != self.start and cell != self.end:
                    if not (self.grid_data and self.grid_data[cell[0]][cell[1]] == 1):
                        if self.terrain_data and self.terrain_data[cell[0]][cell[1]]:
                            base_color = self.terrain_data[cell[0]][cell[1]]['color']
                            self.canvas.itemconfig(self.cells[cell], fill=self.blend_colors(base_color, 'blue'))
                        else:
                            self.canvas.itemconfig(self.cells[cell], fill=self.blend_colors('white', 'blue', alpha=0.5))
                    if self.terrain_data and self.terrain_data[cell[0]][cell[1]]:
                        terrain = self.terrain_data[cell[0]][cell[1]]
                    else:
                        terrain = TERRAIN_TYPES['DEFAULT']
                    fuel_cost = terrain['fuel_cost']
                    self.current_fuel -= fuel_cost
                    self.fuel_consumption_label.configure(
                        text=f'Last Move: {fuel_cost} fuel ({terrain.get("name", "Unknown")})'
                    )
                    if cell in self.fuel_stations:
                        self.current_fuel = min(self.max_fuel, self.current_fuel + 20)
                        stations_visited = int(self.fuel_stations_visited_label.cget("text").split(": ")[1]) + 1
                        self.fuel_stations_visited_label.configure(
                            text=f'Stations Visited: {stations_visited}'
                        )
                    self.update_fuel_display()
                self.path_length_label.configure(text=str(self.path_length))
                self.time_label.configure(text=f"{time.time() - self.start_time:.2f}")
            self.after(self.delay, self.step)
        except StopIteration:
            elapsed = time.time() - self.start_time
            self.visited_label.configure(text=str(self.visited_count))
            self.path_length_label.configure(text=str(self.path_length))
            self.time_label.configure(text=f"{elapsed:.2f}")
            self.remaining_fuel_label.configure(text=f'Remaining Fuel: {self.current_fuel}')
            self.final_fuel_label.configure(text=str(self.current_fuel))
            # Không gọi lại self.after nữa nếu generator đã hết

    def astar_generator(self, grid, start, goal):
        n = len(grid)
        g_score = {start: 0}  # Distance from start
        f_score = {start: logic.get_heuristic(start, goal, 'Manhattan')}  # Use Manhattan for direct paths
        open_set = [(f_score[start], start)]
        came_from = {}
        closed = set()
        fuel_at_node = {start: self.current_fuel}  # Track fuel at each node
        fuel_efficiency = {start: 0}  # Track fuel efficiency at each node
        is_direct_path = all(grid[r][c] == 0 for r in range(n) for c in range(n))
        max_steps = n * n * 2  # Giảm giới hạn số bước duyệt tối đa
        steps = 0
        while open_set:
            steps += 1
            if steps > max_steps:
                print(f"[EcoBot] Pathfinding stopped: exceeded max_steps={max_steps}")
                if hasattr(self, 'time_label'):
                    self.time_label.configure(text='Pathfinding took too long or no path found!')
                return  # Tránh block UI nếu thuật toán chạy quá lâu
            _, current = heapq.heappop(open_set)
            if current == goal:
                break
            if current in closed:
                continue
            closed.add(current)
            yield 'visit', current
            current_fuel = fuel_at_node[current]
            if is_direct_path:
                neighbors = []
                for v, _ in logic.get_neighbors_cost(current, n):
                    if grid[v[0]][v[1]] == 1:
                        continue
                    dx = goal[0] - current[0]
                    dy = goal[1] - current[1]
                    ndx = v[0] - current[0]
                    ndy = v[1] - current[1]
                    direction_score = (dx * ndx + dy * ndy) / (abs(dx) + abs(dy) + 1e-6)
                    neighbors.append((v, direction_score))
                neighbors.sort(key=lambda x: -x[1])
                neighbors = [v for v, _ in neighbors]
            else:
                neighbors = [v for v, _ in logic.get_neighbors_cost(current, n) if grid[v[0]][v[1]] != 1]
            for v in neighbors:
                if not self.terrain_data or not self.terrain_data[v[0]][v[1]]:
                    terrain = TERRAIN_TYPES['DEFAULT']
                else:
                    terrain = self.terrain_data[v[0]][v[1]]
                fuel_cost = terrain['fuel_cost']
                if current_fuel < fuel_cost:
                    continue
                new_fuel = current_fuel - fuel_cost
                if v in self.fuel_stations:
                    new_fuel = min(self.max_fuel, new_fuel + 20)
                distance_cost = 1
                terrain_penalty = 0
                if terrain != TERRAIN_TYPES['DEFAULT']:
                    terrain_penalty = terrain['fuel_cost']
                current_efficiency = new_fuel / (distance_cost + terrain_penalty) if (distance_cost + terrain_penalty) > 0 else 0
                tentative_g = g_score[current] + distance_cost + terrain_penalty
                if current_efficiency > fuel_efficiency.get(current, 0):
                    tentative_g -= current_efficiency * 0.5
                if v == goal:
                    tentative_g -= new_fuel * 0.3
                if v in closed:
                    continue
                if v not in g_score or tentative_g < g_score[v]:
                    came_from[v] = current
                    g_score[v] = tentative_g
                    fuel_at_node[v] = new_fuel
                    fuel_efficiency[v] = current_efficiency
                    if is_direct_path:
                        h_score = logic.get_heuristic(v, goal, 'Manhattan')
                    else:
                        h_score = logic.get_heuristic(v, goal, 'Euclidean')
                        if terrain != TERRAIN_TYPES['DEFAULT']:
                            h_score *= (1 + terrain['fuel_cost'] * 0.1)
                        h_score *= (1 - current_efficiency * 0.2)
                        if new_fuel < self.max_fuel * 0.3:
                            nearest_station_dist = float('inf')
                            for station in self.fuel_stations:
                                dist = logic.get_heuristic(v, station, 'Manhattan')
                                nearest_station_dist = min(nearest_station_dist, dist)
                            if nearest_station_dist != float('inf'):
                                h_score *= (1 + nearest_station_dist * 0.1)
                    f_score[v] = tentative_g + h_score
                    heapq.heappush(open_set, (f_score[v], v))
        # Nếu không tìm được đường đi thì dừng generator ngay
        if goal not in came_from and start != goal:
            return
        # reconstruct path an toàn, tránh vòng lặp vô hạn
        path = []
        node = goal
        found_path = False
        if node == start:
            path.append(start)
            found_path = True
        else:
            visited_reconstruct = set()
            while node != start:
                if node in visited_reconstruct:
                    print("[EcoBot] Infinite loop detected in reconstruct path!")
                    if hasattr(self, 'time_label'):
                        self.time_label.configure(text='Error: Infinite loop in reconstruct path!')
                    return
                visited_reconstruct.add(node)
                path.append(node)
                print(f"[EcoBot] Reconstructing: {node} -> {came_from.get(node)}")
                if node not in came_from:
                    print("[EcoBot] Reconstruct failed: node not in came_from")
                    if hasattr(self, 'time_label'):
                        self.time_label.configure(text='Error: Cannot reconstruct path!')
                    return
                node = came_from[node]
            path.append(start)
            found_path = True
        if not found_path or len(path) == 0:
            print("[EcoBot] No path found or reconstruct failed!")
            if hasattr(self, 'time_label'):
                self.time_label.configure(text='No path found or reconstruct failed!')
            return
        for cell in reversed(path):
            yield 'path', cell

    def _prepend_first(self, first, gen):
        yield first
        yield from gen

if __name__ == '__main__':
    app = EcoBotNavigatorApp()
    app.mainloop() 
