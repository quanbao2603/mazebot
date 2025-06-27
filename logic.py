import random
import heapq
from collections import deque

# --- Tiện ích cho thuật toán tìm đường ---
def get_neighbors(pos, n):
    """Trả về các ô lân cận 4 hướng trong phạm vi lưới."""
    r, c = pos
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            yield (nr, nc)

def get_neighbors_cost(pos, n):
    """Trả về các ô lân cận 8 hướng và chi phí di chuyển."""
    r, c = pos
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n:
            cost = 1 if dr == 0 or dc == 0 else (2**0.5)
            yield (nr, nc), cost

def get_heuristic(a, b, method):
    """Tính heuristic giữa hai điểm a, b theo phương pháp chỉ định."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if method == 'Euclidean':
        return (dx*dx + dy*dy) ** 0.5
    elif method == 'Manhattan':
        return dx + dy
    elif method == 'Chebyshev':
        return max(dx, dy)
    elif method == 'Octile':
        f = 2**0.5 - 1
        return f * min(dx, dy) + abs(dx - dy)
    elif method == 'Tie-breaking':
        return (dx + dy) * (1 + 1e-3)
    elif method == 'Angle Euclidean':
        return (dx*dx + dy*dy) ** 0.5 * (1 + 1e-3)
    return 0

# --- Thuật toán tìm đường ---
def bfs_generator(grid, start, goal):
    n = len(grid)
    visited = {start}
    parent = {start: None}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        yield 'visit', u
        if u == goal:
            break
        for v in get_neighbors(u, n):
            if v not in visited and grid[v[0]][v[1]] == 0:
                visited.add(v)
                parent[v] = u
                queue.append(v)
                yield 'visit', v
    else:
        return
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    for cell in reversed(path):
        yield 'path', cell

def dfs_generator(grid, start, goal):
    n = len(grid)
    visited = set()
    parent = {start: None}
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        yield 'visit', u
        if u == goal:
            break
        for v in get_neighbors(u, n):
            if v not in visited and grid[v[0]][v[1]] == 0:
                parent[v] = u
                stack.append(v)
                yield 'visit', v
    else:
        return
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent.get(node)
    for cell in reversed(path):
        yield 'path', cell

def dijkstra_generator(grid, start, goal):
    n = len(grid)
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
        for v, cost in get_neighbors_cost(u, n):
            if grid[v[0]][v[1]] == 1:
                continue
            nd = d + cost
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(heap, (nd, v))
    if goal not in parent and start != goal:
        return
    path = []
    node = goal
    while node != start:
        path.append(node)
        node = parent[node]
    path.append(start)
    for cell in reversed(path):
        yield 'path', cell

def astar_generator(grid, start, goal, heuristic):
    n = len(grid)
    g_score = {start: 0}
    f_score = {start: get_heuristic(start, goal, heuristic)}
    open_set = [(f_score[start], start)]
    came_from = {}
    closed = set()
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        if current in closed:
            continue
        closed.add(current)
        yield 'visit', current
        for v, cost in get_neighbors_cost(current, n):
            if grid[v[0]][v[1]] == 1:
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
        node = came_from[node]
    path.append(start)
    for cell in reversed(path):
        yield 'path', cell

# Thêm lại các hàm trả về đường đi cuối cùng (dùng cho logic nội bộ)
def bfs(grid, start, goal):
    gen = bfs_generator(grid, start, goal)
    path = []
    for typ, cell in gen:
        if typ == 'path':
            path.append(cell)
    return path

def dfs(grid, start, goal):
    gen = dfs_generator(grid, start, goal)
    path = []
    for typ, cell in gen:
        if typ == 'path':
            path.append(cell)
    return path

def dijkstra(grid, start, goal):
    gen = dijkstra_generator(grid, start, goal)
    path = []
    for typ, cell in gen:
        if typ == 'path':
            path.append(cell)
    return path

def astar(grid, start, goal, heuristic):
    gen = astar_generator(grid, start, goal, heuristic)
    path = []
    for typ, cell in gen:
        if typ == 'path':
            path.append(cell)
    return path

# --- Sinh mê cung ---
def add_loops(grid, n, loops):
    """Thêm các vòng lặp để tạo nhiều đường đi hơn."""
    walls = [(r, c) for r in range(1, n-1) for c in range(1, n-1) if grid[r][c] == 1]
    random.shuffle(walls)
    removed = 0
    for r, c in walls:
        if removed >= loops:
            break
        if (grid[r-1][c] == 0 and grid[r+1][c] == 0) or \
           (grid[r][c-1] == 0 and grid[r][c+1] == 0):
            grid[r][c] = 0
            removed += 1
    return grid

def add_targeted_loops(grid, start, end, loops=1):
    """Thêm vòng lặp dọc theo đường đi duy nhất từ start đến end."""
    path = bfs(grid, start, end)
    if not path:
        return grid
    n = len(grid)
    candidates = []
    for idx, (r, c) in enumerate(path):
        for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                neigh = []
                for dr2, dc2 in [(1,0),(-1,0),(0,1),(0,-1)]:
                    ar, ac = nr + dr2, nc + dc2
                    if (ar, ac) in path:
                        neigh.append(path.index((ar, ac)))
                if len(neigh) >= 2 and abs(neigh[0] - neigh[1]) > 1:
                    candidates.append((nr, nc))
    random.shuffle(candidates)
    for r, c in candidates[:loops]:
        grid[r][c] = 0
    return grid

def maze_recursive_backtracking(n):
    """Sinh mê cung bằng đệ quy quay lui."""
    grid = [[1] * n for _ in range(n)]
    def carve(r, c):
        dirs = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                grid[r + dr//2][c + dc//2] = 0
                grid[nr][nc] = 0
                carve(nr, nc)
    sr = random.randrange(0, n-2, 2)
    sc = random.randrange(0, n-2, 2)
    grid[sr][sc] = 0
    carve(sr, sc)
    return grid

def maze_prim(n):
    """Sinh mê cung bằng thuật toán Prim."""
    grid = [[1] * n for _ in range(n)]
    sr = random.randrange(0, n-2, 2)
    sc = random.randrange(0, n-2, 2)
    grid[sr][sc] = 0
    walls = []
    for dr, dc in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < n and 0 <= nc < n:
            walls.append((nr, nc, (sr, sc)))
    while walls:
        idx = random.randrange(len(walls))
        r, c, (pr, pc) = walls.pop(idx)
        if grid[r][c] == 1 and grid[pr][pc] == 0:
            grid[(r+pr)//2][(c+pc)//2] = 0
            grid[r][c] = 0
            for dr, dc in [(2, 0), (-2, 0), (0, 2), (0, -2)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == 1:
                    walls.append((nr, nc, (r, c)))
    return grid

def maze_kruskal(n):
    """Sinh mê cung bằng thuật toán Kruskal."""
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    cells = [(r, c) for r in range(0, n-1, 2) for c in range(0, n-1, 2)]
    for cell in cells:
        parent[cell] = cell
    edges = []
    for r, c in cells:
        if r + 2 < n:
            edges.append(((r, c), (r+2, c)))
        if c + 2 < n:
            edges.append(((r, c), (r, c+2)))
    random.shuffle(edges)
    grid = [[1] * n for _ in range(n)]
    for r, c in cells:
        grid[r][c] = 0
    for a, b in edges:
        if find(a) != find(b):
            union(a, b)
            ar, ac = a
            br, bc = b
            grid[(ar+br)//2][(ac+bc)//2] = 0
    return grid

def maze_eller(n):
    """Sinh mê cung bằng thuật toán Eller (theo từng hàng)."""
    grid = [[1] * n for _ in range(n)]
    if n < 3:
        return [[0] * n for _ in range(n)]
    cell_rows = [i for i in range(n) if i % 2 == 0]
    cell_cols = [i for i in range(n) if i % 2 == 0]
    sets = {}
    next_set_id = 1
    for row_idx, y in enumerate(cell_rows):
        for x in cell_cols:
            grid[y][x] = 0
            if (y, x) not in sets:
                sets[(y, x)] = next_set_id
                next_set_id += 1
        is_last = (row_idx == len(cell_rows) - 1)
        for i in range(len(cell_cols) - 1):
            x = cell_cols[i]
            x2 = cell_cols[i+1]
            if sets[(y, x)] != sets[(y, x2)]:
                if is_last or random.choice([True, False]):
                    grid[y][x+1] = 0
                    old_id = sets[(y, x2)]
                    new_id = sets[(y, x)]
                    for xc in cell_cols:
                        if sets.get((y, xc)) == old_id:
                            sets[(y, xc)] = new_id
        if not is_last:
            next_y = cell_rows[row_idx + 1]
            groups = {}
            for x in cell_cols:
                sid = sets[(y, x)]
                groups.setdefault(sid, []).append(x)
            new_sets = {}
            for sid, xs in groups.items():
                choices = [x for x in xs if random.choice([True, False])]
                if not choices:
                    choices = [random.choice(xs)]
                for x in choices:
                    grid[y+1][x] = 0
                    grid[next_y][x] = 0
                    new_sets[(next_y, x)] = sid
            for x in cell_cols:
                key = (next_y, x)
                if key not in new_sets:
                    new_sets[key] = next_set_id
                    next_set_id += 1
                    grid[next_y][x] = 0
            sets = new_sets
    return grid

# --- Bảng ánh xạ các thuật toán sinh mê cung ---
MAZE_GENERATORS = {
    'Recursive Backtracking': maze_recursive_backtracking,
    'Prim': maze_prim,
    'Kruskal': maze_kruskal,
    'Eller': maze_eller
}

def generate_maze(n, algorithm='Recursive Backtracking', variant=None):
    """
    Sinh mê cung kích thước n x n.
    algorithm: ['Recursive Backtracking','Prim','Kruskal','Eller'].
    variant: tuỳ chọn biến thể (chưa dùng).
    Trả về lưới 2D với 0=lối đi, 1=tường.
    """
    try:
        gen_func = MAZE_GENERATORS[algorithm]
    except KeyError:
        raise ValueError(f'Unknown algorithm: {algorithm}')
    grid = gen_func(n)
    loops = max(1, n // 10)
    grid = add_loops(grid, n, loops)
    return grid