import customtkinter as ctk
import tkinter as tk
from collections import deque
import random
import logic as logic
import time  # for measuring elapsed time
import subprocess, sys, os
from PIL import Image, ImageDraw, ImageTk  # for creating icons
from logic import bfs_generator, dfs_generator, dijkstra_generator, astar_generator

# Configure CustomTkinter
ctk.set_appearance_mode('light')
ctk.set_default_color_theme('green')

class PathVisualizerApp(ctk.CTk):
    # --- Khởi tạo và UI ---
    def __init__(self):
        super().__init__()
        self.title('WOM MAZE SOLVER')
        # Start maximized and allow resizing
        self.state('zoomed')
        self.resizable(True, True)
        self.grid_size = 50
        self.grid_data = None
        # default weight for weighted A*
        self.weight = 2
        # Initialize icons
        self.start_icon = None
        self.end_icon = None

        # Control panel (left) with increased width
        control = ctk.CTkFrame(self, width=300, fg_color='#00a000') 
        control.grid(row=0, column=0, sticky='ns')
        control.grid_propagate(False)
        control.grid_columnconfigure(0, weight=1)

        # Title
        ctk.CTkLabel(control, text='WOM MAZE', font=('Arial', 18, 'bold'), text_color='white').grid(row=0, column=0, pady=(20,10), padx=10)

        # Pathfinding Algorithm
        ctk.CTkLabel(control, text='Pathfinding Algorithm', text_color='white').grid(row=1, column=0, pady=(10,2), padx=10)
        self.combo_algo = ctk.CTkComboBox(control, width=260, values=['BFS','DFS','Dijkstra','A*'])
        self.combo_algo.set('BFS')
        self.combo_algo.grid(row=2, column=0, padx=10, pady=(0,10))
        self.combo_algo.configure(command=self.on_algo_change)

        # Heuristic selector
        ctk.CTkLabel(control, text='Heuristic', text_color='white').grid(row=3, column=0, pady=(10,2), padx=10)
        self.combo_heur = ctk.CTkComboBox(control, width=260, values=['Euclidean','Manhattan','Chebyshev','Octile','Squared Euclidean'])
        self.combo_heur.set('Euclidean')
        self.combo_heur.grid(row=4, column=0, padx=10, pady=(0,10))
        self.combo_heur.configure(state='disabled')

        # Maze Generation
        ctk.CTkLabel(control, text='Maze Generation', text_color='white').grid(row=5, column=0, pady=(10,2), padx=10)
        self.combo_maze = ctk.CTkComboBox(control, width=260, values=['Recursive Backtracking','Prim','Kruskal','Eller'])
        self.combo_maze.set('Recursive Backtracking')
        self.combo_maze.grid(row=6, column=0, padx=10, pady=(0,10))
        # Special Maze Variant
        ctk.CTkLabel(control, text='Maze Variant', text_color='white').grid(row=7, column=0, pady=(10,2), padx=10)
        self.combo_variant = ctk.CTkComboBox(control, width=260, values=['EcoBot Navigator','Mud Maze','Zero','Compare'])
        self.combo_variant.set('EcoBot Navigator')
        self.combo_variant.grid(row=8, column=0, padx=10, pady=(0,10))
        # bind maze variant change to toggle New Maze Window button
        self.combo_variant.configure(command=self.on_variant_change)

        # Buttons
        self.btn_generate = ctk.CTkButton(control, width=260, text='Generate Maze', corner_radius=10,
                                          fg_color='white', text_color='black', hover_color='red', command=self.on_generate)
        self.btn_generate.grid(row=9, column=0, padx=10, pady=(20,5))
        self.btn_start = ctk.CTkButton(control, width=260, text='Start', corner_radius=10,
                                       fg_color='white', text_color='black', hover_color='red', command=self.start_pathfinding)
        self.btn_start.grid(row=10, column=0, padx=10, pady=5)
        self.btn_reset = ctk.CTkButton(control, width=260, text='Reset', corner_radius=10,
                                       fg_color='white', text_color='black', hover_color='red', command=self.on_reset)
        self.btn_reset.grid(row=11, column=0, padx=10, pady=5)

        # Grid size slider (max 100)
        ctk.CTkLabel(control, text='Grid Size', text_color='white').grid(row=12, column=0, pady=(20,2), padx=10)
        self.slider = ctk.CTkSlider(control, width=260, from_=20, to=100, command=self.update_grid)
        self.slider.set(self.grid_size)
        self.slider.grid(row=13, column=0, padx=10, pady=(0,5))
        self.lbl_size = ctk.CTkLabel(control, text=str(self.grid_size), text_color='white')
        self.lbl_size.grid(row=14, column=0, padx=10)

        # Apply grid size
        self.btn_apply = ctk.CTkButton(control, width=260, text='Apply Grid Size', corner_radius=10,
                                      fg_color='white', text_color='black', hover_color='red', command=self.apply_grid)
        self.btn_apply.grid(row=15, column=0, padx=10, pady=(10,5))
        # Move speed slider (integers 1-10)
        self.speed_slider = ctk.CTkSlider(control, width=260, from_=1, to=10, number_of_steps=9, command=self.change_move_speed)
        self.speed_slider.set(5)
        self.speed_slider.grid(row=16, column=0, padx=10, pady=(0,5))
        self.lbl_speed = ctk.CTkLabel(control, text='5', text_color='white')
        self.lbl_speed.grid(row=17, column=0, padx=10)
        # initialize delay based on default speed after label exists
        self.change_move_speed(self.speed_slider.get())
        # Apply Move Speed button
        self.btn_apply_speed = ctk.CTkButton(control, width=260, text='Apply Move Speed', corner_radius=10,
                                            fg_color='white', text_color='black', hover_color='red', command=self.apply_speed)
        self.btn_apply_speed.grid(row=18, column=0, padx=10, pady=(5,10))

        # Canvas area
        canvas_frame = ctk.CTkFrame(self, fg_color='white')
        canvas_frame.grid(row=0, column=1, sticky='nsew')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind('<Configure>', lambda e: self.draw_grid(self.grid_size))
        # Initialize start/end selection and grid metrics
        self.start = None
        self.end = None
        self.cell_size = None
        self.pad_x = None
        self.pad_y = None
        self.canvas.bind('<Button-1>', self.on_canvas_left_click)
        self.canvas.bind('<Button-3>', self.on_canvas_right_click)

        # Additional right control panel
        right_control = ctk.CTkFrame(self, width=300, fg_color='#00a000')
        right_control.grid(row=0, column=2, sticky='ns')
        right_control.grid_propagate(False)
        right_control.grid_columnconfigure(0, weight=1)
        # Extra Controls Header
        ctk.CTkLabel(right_control, text='Extra Controls', text_color='white', font=('Arial', 16, 'bold')).grid(row=0, column=0, pady=10, padx=10)
        # Metrics panel with styled frame
        metrics_frame = ctk.CTkFrame(right_control, fg_color='#004d00', corner_radius=10)
        metrics_frame.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        metrics_frame.grid_columnconfigure(1, weight=1)
        # Visited Nodes
        ctk.CTkLabel(metrics_frame, text='Visited Nodes:', text_color='white', font=('Arial', 12, 'bold'))\
            .grid(row=0, column=0, sticky='w', padx=5, pady=(5,2))
        self.lbl_visited = ctk.CTkLabel(metrics_frame, text='0', text_color='white', font=('Arial', 12))
        self.lbl_visited.grid(row=0, column=1, sticky='e', padx=5, pady=(5,2))
        # Separator line
        sep1 = ctk.CTkFrame(metrics_frame, height=1, fg_color='white')
        sep1.grid(row=1, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        # Path Length
        ctk.CTkLabel(metrics_frame, text='Path Length:', text_color='white', font=('Arial', 12, 'bold'))\
            .grid(row=2, column=0, sticky='w', padx=5, pady=(2,2))
        self.lbl_path_length = ctk.CTkLabel(metrics_frame, text='0', text_color='white', font=('Arial', 12))
        self.lbl_path_length.grid(row=2, column=1, sticky='e', padx=5, pady=(2,2))
        # Separator line
        sep2 = ctk.CTkFrame(metrics_frame, height=1, fg_color='white')
        sep2.grid(row=3, column=0, columnspan=2, sticky='ew', padx=5, pady=2)
        # Time Elapsed
        ctk.CTkLabel(metrics_frame, text='Time Elapsed (s):', text_color='white', font=('Arial', 12, 'bold'))\
            .grid(row=4, column=0, sticky='w', padx=5, pady=(2,5))
        self.lbl_time = ctk.CTkLabel(metrics_frame, text='0.00', text_color='white', font=('Arial', 12))
        self.lbl_time.grid(row=4, column=1, sticky='e', padx=5, pady=(2,5))
        # add New Maze Window button (enabled only when variant != 'Zero')
        self.btn_new_window = ctk.CTkButton(right_control, width=260, text='New Maze Window', corner_radius=10,
                                           fg_color='white', text_color='black', hover_color='red',
                                           command=self.on_new_window)
        self.btn_new_window.grid(row=2, column=0, padx=10, pady=(10,5))
        # set initial button state
        self.on_variant_change()

    # --- Xử lý sự kiện UI ---
    def on_algo_change(self, choice):
        if self.combo_algo.get() == 'A*':
            self.combo_heur.configure(state='normal')
        else:
            self.combo_heur.configure(state='disabled')

    def update_grid(self, val):
        self.grid_size = int(val)
        self.lbl_size.configure(text=str(self.grid_size))

    def change_move_speed(self, val):
        """Change move speed"""
        self.speed = val
        self.lbl_speed.configure(text=str(self.speed))
        # compute animation delay: lower delay = faster speed
        self.delay = int(200 / self.speed)

    def apply_speed(self):
        """Apply the selected move speed."""
        # Here you can integrate move speed into maze navigation logic
        print(f"Move speed applied: {self.speed}")
    
    def apply_grid(self):
        self.draw_grid(self.grid_size)

    def create_start_icon(self, size):
        """Create a start icon (green flag)."""
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw flag pole
        pole_width = size // 8
        draw.rectangle([(size//4, size//4), (size//4 + pole_width, size*3//4)], fill='black')
        # Draw flag
        points = [(size//4 + pole_width, size//4), 
                 (size*3//4, size//2),
                 (size//4 + pole_width, size*3//4)]
        draw.polygon(points, fill='green')
        return ImageTk.PhotoImage(img)

    def create_end_icon(self, size):
        """Create an end icon (red target)."""
        img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        # Draw concentric circles
        center = size // 2
        for r in range(size//2, 0, -size//6):
            draw.ellipse([(center-r, center-r), (center+r, center+r)], 
                        outline='red', width=2)
        # Draw center dot
        draw.ellipse([(center-size//8, center-size//8), 
                     (center+size//8, center+size//8)], 
                    fill='red')
        return ImageTk.PhotoImage(img)

    def draw_grid(self, n): 
        """Draw n x n grid with square cells and decorate margins."""
        # Delete everything including start/end icons
        self.canvas.delete('all')
        
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        # determine square cell size
        cell = min(w / n, h / n)
        total = cell * n
        # padding to center grid
        pad_x = (w - total) / 2
        pad_y = (h - total) / 2
        # decorate leftover areas with light green
        margin_color = '#b0e0a0'
        # left margin
        self.canvas.create_rectangle(0, 0, pad_x, h, fill=margin_color, outline='')
        # right margin
        self.canvas.create_rectangle(pad_x + total, 0, w, h, fill=margin_color, outline='')
        # top margin
        self.canvas.create_rectangle(pad_x, 0, pad_x + total, pad_y, fill=margin_color, outline='')
        # bottom margin
        self.canvas.create_rectangle(pad_x, pad_y + total, pad_x + total, h, fill=margin_color, outline='')
        # create cell rectangles and store references
        self.cells = {}
        for row in range(n):
            for col in range(n):
                x1 = pad_x + col * cell
                y1 = pad_y + row * cell
                x2 = x1 + cell
                y2 = y1 + cell
                # Set initial color based on grid_data if it exists
                fill_color = 'black' if self.grid_data and self.grid_data[row][col] == 1 else 'white'
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline='gray')
                self.cells[(row, col)] = rect
        # store grid metrics for click mapping
        self.cell_size = cell
        self.pad_x = pad_x
        self.pad_y = pad_y
        # ensure start/end are within grid after resizing
        if self.start not in self.cells:
            self.start = (0, 0)
        if self.end not in self.cells:
            self.end = (n - 1, n - 1)
            
        # Draw start icon (flag)
        if self.start in self.cells:
            x = pad_x + self.start[1] * cell
            y = pad_y + self.start[0] * cell
            # Draw flag pole
            self.canvas.create_line(x + cell*0.2, y + cell*0.2, 
                                  x + cell*0.2, y + cell*0.8, 
                                  fill='black', width=2, tags='start_icon')
            # Draw flag
            points = [
                x + cell*0.2, y + cell*0.2,  # pole top
                x + cell*0.7, y + cell*0.4,  # flag tip
                x + cell*0.2, y + cell*0.6   # pole middle
            ]
            self.canvas.create_polygon(points, fill='green', outline='black', tags='start_icon')
            
        # Draw end icon (target)
        if self.end in self.cells:
            x = pad_x + self.end[1] * cell
            y = pad_y + self.end[0] * cell
            center_x = x + cell/2
            center_y = y + cell/2
            
            # Draw 3 concentric circles
            radius = cell * 0.4
            for i in range(3):
                r = radius * (1 - i * 0.2)
                if r > 1:  # Only draw if radius is greater than 1 pixel
                    self.canvas.create_oval(center_x-r, center_y-r, 
                                         center_x+r, center_y+r, 
                                         outline='red', width=2, tags='end_icon')
            
            # Draw center dot
            dot_radius = max(2, cell * 0.1)
            self.canvas.create_oval(center_x-dot_radius, center_y-dot_radius,
                                  center_x+dot_radius, center_y+dot_radius,
                                  fill='red', outline='red', tags='end_icon')

    def pixel_to_cell(self, x, y):
        """Convert canvas x,y to grid cell indices."""
        if self.cell_size is None:
            return None, None
        col = int((x - self.pad_x) / self.cell_size)
        row = int((y - self.pad_y) / self.cell_size)
        return row, col

    def on_canvas_left_click(self, event):
        """Handle left-click to set start point."""
        r, c = self.pixel_to_cell(event.x, event.y)
        if r is None or not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return
        # do not allow setting start on a wall
        if self.grid_data is not None and self.grid_data[r][c] == 1:
            return
            
        # Delete all existing start icons first
        self.canvas.delete('start_icon')
        
        # reset previous start (if different from end)
        if self.start is not None and self.start != self.end:
            # Restore original cell color
            self.canvas.itemconfig(self.cells[self.start], 
                                 fill='black' if self.grid_data and self.grid_data[self.start[0]][self.start[1]] == 1 else 'white')
        
        self.start = (r, c)
        # Update start position without redrawing entire grid
        if self.start in self.cells:
            x = self.pad_x + self.start[1] * self.cell_size
            y = self.pad_y + self.start[0] * self.cell_size
            # Clear previous start icon area
            self.canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, 
                                      fill='white', outline='gray', tags='start_icon')
            # Draw flag pole
            self.canvas.create_line(x + self.cell_size*0.2, y + self.cell_size*0.2, 
                                  x + self.cell_size*0.2, y + self.cell_size*0.8, 
                                  fill='black', width=2, tags='start_icon')
            # Draw flag
            points = [
                x + self.cell_size*0.2, y + self.cell_size*0.2,  # pole top
                x + self.cell_size*0.7, y + self.cell_size*0.4,  # flag tip
                x + self.cell_size*0.2, y + self.cell_size*0.6   # pole middle
            ]
            self.canvas.create_polygon(points, fill='green', outline='black', tags='start_icon')

    def on_canvas_right_click(self, event):
        """Handle right-click to set end point."""
        r, c = self.pixel_to_cell(event.x, event.y)
        if r is None or not (0 <= r < self.grid_size and 0 <= c < self.grid_size):
            return
        # do not allow setting end on a wall
        if self.grid_data is not None and self.grid_data[r][c] == 1:
            return
            
        # Delete all existing end icons first
        self.canvas.delete('end_icon')
        
        # reset previous end (if different from start)
        if self.end is not None and self.end != self.start:
            # Restore original cell color
            self.canvas.itemconfig(self.cells[self.end], 
                                 fill='black' if self.grid_data and self.grid_data[self.end[0]][self.end[1]] == 1 else 'white')
        
        self.end = (r, c)
        # Update end position without redrawing entire grid
        if self.end in self.cells:
            x = self.pad_x + self.end[1] * self.cell_size
            y = self.pad_y + self.end[0] * self.cell_size
            # Clear previous end icon area
            self.canvas.create_rectangle(x, y, x + self.cell_size, y + self.cell_size, 
                                      fill='white', outline='gray', tags='end_icon')
            center_x = x + self.cell_size/2
            center_y = y + self.cell_size/2
            
            # Draw 3 concentric circles
            radius = self.cell_size * 0.4
            for i in range(3):
                r = radius * (1 - i * 0.2)
                if r > 1:  # Only draw if radius is greater than 1 pixel
                    self.canvas.create_oval(center_x-r, center_y-r, 
                                         center_x+r, center_y+r, 
                                         outline='red', width=2, tags='end_icon')
            
            # Draw center dot
            dot_radius = max(2, self.cell_size * 0.1)
            self.canvas.create_oval(center_x-dot_radius, center_y-dot_radius,
                                  center_x+dot_radius, center_y+dot_radius,
                                  fill='red', outline='red', tags='end_icon')

    def on_generate(self):
        """Generate a maze and draw walls."""
        n = self.grid_size
        
        # Reset start and end points
        self.start = None
        self.end = None
        # Delete all existing start/end icons
        self.canvas.delete('start_icon')
        self.canvas.delete('end_icon')
        
        # Generate new maze
        self.grid_data = logic.generate_maze(n, self.combo_maze.get(), self.combo_variant.get())
        
        # Draw the maze
        for (r, c), rect in self.cells.items():
            color = 'black' if self.grid_data[r][c] == 1 else 'white'
            self.canvas.itemconfig(rect, fill=color)
            
        # Reset metrics
        self.visited_count = 0
        self.path_length = 0
        self.lbl_visited.configure(text='0')
        self.lbl_path_length.configure(text='0')
        self.lbl_time.configure(text='0.00')
        
        # Vẽ lại các icon start và end
        if self.start:
            x = self.pad_x + self.start[1] * self.cell_size
            y = self.pad_y + self.start[0] * self.cell_size
            # Draw flag pole
            self.canvas.create_line(x + self.cell_size*0.2, y + self.cell_size*0.2, 
                                  x + self.cell_size*0.2, y + self.cell_size*0.8, 
                                  fill='black', width=2)
            # Draw flag
            points = [
                x + self.cell_size*0.2, y + self.cell_size*0.2,  # pole top
                x + self.cell_size*0.7, y + self.cell_size*0.4,  # flag tip
                x + self.cell_size*0.2, y + self.cell_size*0.6   # pole middle
            ]
            self.canvas.create_polygon(points, fill='green', outline='black')
            
        if self.end:
            x = self.pad_x + self.end[1] * self.cell_size
            y = self.pad_y + self.end[0] * self.cell_size
            center_x = x + self.cell_size/2
            center_y = y + self.cell_size/2
            # Draw concentric circles
            for r in range(int(self.cell_size*0.4), 0, -int(self.cell_size*0.1)):
                self.canvas.create_oval(center_x-r, center_y-r, 
                                     center_x+r, center_y+r, 
                                     outline='red', width=2)
            # Draw center dot
            self.canvas.create_oval(center_x-self.cell_size*0.1, center_y-self.cell_size*0.1,
                                  center_x+self.cell_size*0.1, center_y+self.cell_size*0.1,
                                  fill='red', outline='red')

    def start_pathfinding(self):
        self.visited_count = 0
        self.path_length = 0
        self.start_time = time.time()
        self.stepping = True
        try:
            n = self.grid_size
            # Kiểm tra hợp lệ điểm bắt đầu và kết thúc
            if self.start is None or self.end is None:
                self.lbl_time.configure(text='Please select start and end points!')
                self.lbl_visited.configure(text='0')
                self.lbl_path_length.configure(text='0')
                return
            if self.grid_data is not None:
                if self.grid_data[self.start[0]][self.start[1]] == 1 or self.grid_data[self.end[0]][self.end[1]] == 1:
                    self.lbl_time.configure(text='Start/End cannot be on a wall!')
                    self.lbl_visited.configure(text='0')
                    self.lbl_path_length.configure(text='0')
                    return
            for (r, c), rect in self.cells.items():
                if self.grid_data is not None:
                    color = 'black' if self.grid_data[r][c] == 1 else 'white'
                else:
                    color = 'white'
                self.canvas.itemconfig(rect, fill=color)
            self.draw_grid(self.grid_size)
            grid = self.grid_data if self.grid_data is not None else [[0] * n for _ in range(n)]
            algo = self.combo_algo.get()
            if algo == 'BFS':
                self.step_gen = bfs_generator(grid, self.start, self.end)
            elif algo == 'DFS':
                self.step_gen = dfs_generator(grid, self.start, self.end)
            elif algo == 'Dijkstra':
                self.step_gen = dijkstra_generator(grid, self.start, self.end)
            else:
                heur = self.combo_heur.get()
                self.step_gen = astar_generator(grid, self.start, self.end, heur)
            self.after(self.delay, self.step)
        except Exception as e:
            self.lbl_time.configure(text=f'Error: {e}')

    def on_reset(self):
        # Dừng animation nếu đang dò đường
        self.stepping = False
        self.grid_data = None
        self.start = None
        self.end = None
        self.draw_grid(self.grid_size)

    def get_weighted_path(self, grid, start, goal, heuristic):
        """Run weighted A* to get full path ignoring visit steps."""
        path = []
        for typ, cell in self.astar_generator(grid, start, goal, heuristic):
            if typ == 'path':
                path.append(cell)
        return path

    def on_variant_change(self, choice=None):
        """Enable or disable New Maze Window button based on selected variant."""
        if self.combo_variant.get() == 'Zero':
            self.btn_new_window.configure(state='disabled')
        else:
            self.btn_new_window.configure(state='normal')

    def on_new_window(self):
        """Open a new maze solver window as a separate process based on selected variant."""
        variant = self.combo_variant.get()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        mapping = {
            'EcoBot Navigator': 'WOM_MAZE_ECOBOT_UI.py',
            'Mud Maze': 'WOM_MAZE_MUD_UI.py',
            'Compare': 'WOM_MAZE_COMPARE_UI.py'
        }
        script_name = mapping.get(variant)
        if script_name:
            script_path = os.path.join(script_dir, script_name)
            subprocess.Popen([sys.executable, script_path])

    def step(self):
        if not getattr(self, 'stepping', True):
            return
        try:
            typ, cell = next(self.step_gen)
            if typ == 'visit':
                self.visited_count += 1
                if cell != self.start and cell != self.end:
                    self.canvas.itemconfig(self.cells[cell], fill='#ADD8E6')
                self.lbl_visited.configure(text=str(self.visited_count))
                self.lbl_time.configure(text=f"{time.time() - self.start_time:.2f}")
            else:
                self.path_length += 1
                if cell != self.start and cell != self.end:
                    self.canvas.itemconfig(self.cells[cell], fill='blue')
                self.lbl_path_length.configure(text=str(self.path_length))
                self.lbl_time.configure(text=f"{time.time() - self.start_time:.2f}")
            self.after(self.delay, self.step)
        except StopIteration:
            elapsed = time.time() - self.start_time
            self.lbl_visited.configure(text=str(self.visited_count))
            self.lbl_path_length.configure(text=str(self.path_length))
            self.lbl_time.configure(text=f"{elapsed:.2f}")
            self.stepping = False
            return

if __name__ == '__main__':
    app = PathVisualizerApp()
    app.mainloop() 