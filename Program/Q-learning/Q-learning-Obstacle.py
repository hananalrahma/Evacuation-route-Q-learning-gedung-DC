# jalan
import tkinter as tk
import random
import matplotlib.pyplot as plt
import threading
import time

# Ukuran grid
grid_width = 50
grid_height = 20

# Inisialisasi lingkungan grid
environment = [[0] * grid_width for _ in range(grid_height)]

# Inisialisasi posisi awal dan posisi tujuan
start_position = (0, 0)
goal_position = (grid_height - 1, grid_width - 1)

# Fungsi untuk mengatur obstacle
def set_obstacle(row, col):
    environment[row][col] = 1

# Fungsi untuk mengatur posisi awal agen
def set_start_position(row, col):
    global start_position
    start_position = (row, col)
    draw_agent_and_goal()

# Fungsi untuk mengatur posisi tujuan agen
def set_goal_position(row, col):
    global goal_position
    goal_position = (row, col)
    draw_agent_and_goal()

# Inisialisasi Q-table
q_table = [[0] * 4 for _ in range(grid_width * grid_height)]

def get_action(state):
    # Memilih aksi dengan epsilon-greedy policy
    epsilon = 0.1
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return q_table[state].index(max(q_table[state]))

def get_reward(state, environment):
    # Mendapatkan reward berdasarkan lingkungan
    row = state // grid_width
    col = state % grid_width
    if environment[row][col] == 1:
        return -10  # Jika obstacle
    elif row == goal_position[0] and col == goal_position[1]:
        return 10  # Jika mencapai tujuan
    else:
        return -1  # Jika langkah normal
    

def update_q_table(state, action, next_state, reward):
    # Memperbarui nilai Q berdasarkan algoritma Q-learning
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    q_table[state][action] += alpha * (reward + gamma * max(q_table[next_state]) - q_table[state][action])

def find_shortest_path():
    # Mencari rute terpendek berdasarkan Q-table
    shortest_path = []
    state = start_position[0] * grid_width + start_position[1]
    shortest_path.append(state)

    while state != goal_position[0] * grid_width + goal_position[1]:
        action = q_table[state].index(max(q_table[state]))
        row = state // grid_width
        col = state % grid_width

        if action == 0:  # Up
            row = max(row - 1, 0)
        elif action == 1:  # Down
            row = min(row + 1, grid_height - 1)
        elif action == 2:  # Left
            col = max(col - 1, 0)
        elif action == 3:  # Right
            col = min(col + 1, grid_width - 1)

        state = row * grid_width + col
        shortest_path.append(state)
        

    return shortest_path

# Menghitung jumlah langkah terpendek
def display_shortest_path_length():
    shortest_path = find_shortest_path()
    shortest_path_length = len(shortest_path) - 1
    
    return shortest_path_length
    

# Menampilkan Q-table untuk jalur terpendek
def display_shortest_q_table():
    # Menampilkan nilai Q-table
    print("Q-Table Shortest Path:")
    for state in find_shortest_path():
        print(q_table[state])

# Membuat GUI Tkinter
root = tk.Tk()
canvas = tk.Canvas(root, width=1000, height=400)
canvas.pack()

def draw_environment(environment):
    # Menggambar lingkungan grid
    canvas.delete("all")
    cell_width = 20

    for row in range(grid_height):
        for col in range(grid_width):
            x1 = col * cell_width
            y1 = row * cell_width
            x2 = x1 + cell_width
            y2 = y1 + cell_width

            if environment[row][col] == 1:
                canvas.create_rectangle(x1, y1, x2, y2, fill="black")

            canvas.create_line(x1, y1, x2, y1)
            canvas.create_line(x1, y1, x1, y2)
    root.update()

def draw_agent_and_goal():
    # Menggambar posisi awal agen dan posisi tujuan agen
    cell_width = 20
    start_row, start_col = start_position
    goal_row, goal_col = goal_position

    x1 = start_col * cell_width + cell_width // 2
    y1 = start_row * cell_width + cell_width // 2
    canvas.delete("agent")
    canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, fill="#254ED7", tags="agent")

    x2 = goal_col * cell_width + cell_width // 2
    y2 = goal_row * cell_width + cell_width // 2
    canvas.delete("goal")
    canvas.create_rectangle(x2 - 10, y2 - 10, x2 + 10, y2 + 10, fill="#1ACB26", tags="goal")

    root.update()

def check_convergence(episode_rewards):
    # Mengecek konvergensi berdasarkan perbedaan rewards episode sebelumnya dan saat ini
    if len(episode_rewards) > 1:
        last_reward = episode_rewards[-2]
        current_reward = episode_rewards[-1]
        diff = abs(current_reward - last_reward)
        if diff < 0.1:
            return True
    return False

def train_q_learning():
    # Melatih Q-learning untuk mencari jalur terpendek
    num_episodes = 3000
    episode_rewards = []
    episode_steps = []
    converged_episode = -1
    
    start_time = time.process_time()

    for episode in range(num_episodes):
        state = start_position[0] * grid_width + start_position[1]  # Posisi awal
        done = False
        episode_reward = 0
        episode_step = 0

        while not done:
            action = get_action(state)
            row = state // grid_width
            col = state % grid_width

            if action == 0:  # Up
                row = max(row - 1, 0)
            elif action == 1:  # Down
                row = min(row + 1, grid_height - 1)
            elif action == 2:  # Left
                col = max(col - 1, 0)
            elif action == 3:  # Right
                col = min(col + 1, grid_width - 1)

            next_state = row * grid_width + col
            reward = get_reward(next_state, environment)
            update_q_table(state, action, next_state, reward)
            state = next_state

            episode_reward += reward
            episode_step += 1

            # Cek apakah mencapai tujuan
            if row == goal_position[0] and col == goal_position[1]:
                done = True

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        if check_convergence(episode_rewards) and converged_episode == -1:
            converged_episode = episode
            end_time = time.process_time()
            elapsed_time = end_time - start_time
            print("Time to Convergence:", elapsed_time, "seconds")
        
    # Menampilkan Q-table setelah training selesai
    # display_shortest_q_table()

    # Menampilkan rute terpendek
    shortest_path = find_shortest_path()
    print("Shortest Path:")
    print(shortest_path)
    
    shortest_path_length = display_shortest_path_length()
    print("Shortest Path Length:", shortest_path_length)

    print("Converged Episode:", converged_episode)
    # Menggambar rute terpendek
    def draw_shortest_path():
        canvas.delete("path")
        cell_width = 20
        for state in shortest_path:
            row = state // grid_width
            col = state % grid_width
            x1 = col * cell_width + cell_width // 2
            y1 = row * cell_width + cell_width // 2
            canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, fill="blue", tags="path")
        root.update()

    draw_shortest_path()

    # Mengukur CPU time
    end_time = time.process_time()
    cpu_time = end_time - start_time
    print("CPU Time:", cpu_time, "seconds")
    
    #grafik terpisah 1
    plt.figure()
    plt.plot(range(num_episodes), episode_rewards, 'b')
    plt.title('Episode via rewards')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    
    #grafik terpisah 2
    plt.figure()
    plt.plot(range(num_episodes), episode_steps, 'r')
    plt.title('Episode via steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    
    plt.show()
    
     
def start_training():
    # Memulai proses pelatihan dalam thread terpisah
    training_thread = threading.Thread(target=train_q_learning)
    training_thread.start()


# Mengatur obstacle
obstacles = [
    (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), #tembok tengah antara toilet pria dan wanita.
    (3, 4), (5, 2), (5, 1), (5, 0), (5, 4),
    (5, 5), (5, 6), (5, 7), (0, 7), (1, 7),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
    (7, 7), (7, 7), (9, 7), (10, 7), (10, 6),
    (10, 8), (8, 7), (8, 6), (8, 5), (7, 4), (8, 4), (10, 11),
    (9, 4), (10, 4), (7, 2), (8, 2), (9, 2),
    (10, 2), (10, 2), (10, 1), (10, 0),
    (18, 5), (17, 5), (16, 5), (15, 5), (14, 5),
    (14, 5), (14, 6), (14, 7), (14, 8), (14, 9),
    (14, 10) ,(14, 11), (14, 12), (14, 14), (14, 15),
    (14, 15), (15, 15), (16, 15), (17, 15), (18, 15),
    (14, 16), (14, 19), (14, 20), (14, 21), (14, 22),
    (14, 23), (14, 24), (14, 27), (14, 28), (14, 28),
    (14, 28), (15, 28), (16, 28), (17, 28), (18, 28),
    (15, 30), (15, 31), (15, 32), (15, 33), (15, 34),
    (15, 35), (16, 35), (17, 35), (18, 35),
    (10, 12), (10, 13), (10, 14), (10, 15), (10, 16),
    (10, 17), (10, 18), (10, 19), (10, 21), 
    (10, 20), (9, 20), (8, 20), (7, 20), (6, 20), (5, 20), (4, 20), (3, 20), (2, 20), (1, 20), (0, 20),
    (10, 22), (10, 23), (10, 24), (10, 25), (10, 26),
    (10, 27), (10, 28), (10, 34),
    (10, 34), (10, 35), (10, 36), (10, 37), (10, 38),
    (10, 39), (10, 40), (9, 46), (10, 45),
    (8, 46), (7, 46), (6, 46), (5, 46), (4, 46), (3, 46),
    (2, 46), (1, 46), (0, 46), (9, 33), (8, 33), (7, 33),
    (6, 33), (5, 33), (4, 33), (3, 33), (2, 33), (1, 33),
    (0, 33), (10, 32), (19, 5), (19, 15), (19, 28), (19, 35), (10, 20),
    (10, 29), (10, 33), (10, 41), (10, 42), (10, 46),# Tambahkan obstacle lainnya di sini
    (8, 34), (8, 35), (8, 36), (9, 34), (9, 35), (9, 36), #ruang 1A
    (1, 38), (2, 38), (3, 38), (4, 38), (5, 38), (6, 38),
    (1, 39), (2, 39), (3, 39), (4, 39), (5, 39), (6, 39),
    (4, 41), (4, 42), (4, 43), (4, 44), (4, 45), (6, 42), (7, 42), (8, 42),
    (6, 45), (7, 45), (8, 45), (1, 35), (2, 35), (3, 35), (4, 35), (5, 35), (6, 35), (6, 34), #ruang 1A
    
    (2, 21), (2, 22), (2, 23), (2, 25), (3, 25), (4, 25), (5, 25), (6, 25), #ruang 1B
    (6, 24), (7, 22), (8, 22), (0, 28), (1, 28), (2, 28), (3, 28), (4, 28),
    (0, 29), (1, 29), (2, 29), (3, 29), (4, 29), (7, 27), (7, 28), (7, 29), (7, 30),#ruang 1B
    
    (1, 11), (2, 11), (3, 11), (3, 10), (3, 9), (4, 14), (3, 14), (2, 14),
    (2, 15), (2, 16), (2, 17), (4, 17), (5, 17), (6, 17), (7, 17), (8, 17),
    (4, 18), (5, 18), (6, 18), (7, 18), (8, 18), (6, 8), (6, 9), (6, 10), (6, 11),
    (6, 12), (6, 13), (6, 14), (7, 14), (8, 14),#ruang 1C
    
    (16, 18), (16, 19), (16, 20), (16, 21), (16, 22), (16, 23), (16, 24), (16, 25), (16, 26), (17, 26),
    (18, 18), (18, 19), (18, 20), (18, 21), (18, 22), (18, 23), (18, 24), (18, 25), (18, 26),#ruang rapat 1
    
    (16, 7), (17, 7), (18, 7), (16, 9), (16, 10), (16, 11), (16, 12), (16, 13), (18, 9), (18, 10), (18, 11),
    (18, 12), (18, 13)#ruang rapat 2
]

for obstacle in obstacles:
    row, col = obstacle
    if row < grid_height and col < grid_width:
        set_obstacle(row, col)

draw_environment(environment)
draw_agent_and_goal()

# Mengatur posisi awal agen
set_start_position(1, 34)
# daftar koordinat awal agen:
# 1A: 1,34
# 1B: 1,22
# 1C: 1,9
# ruang rapat 1: 19,27
# ruang rapat 2: 19,6


# Mengatur posisi tujuan agen
set_goal_position(grid_height - 8, grid_width - 50)

# Tombol untuk memulai pelatihan
start_button = tk.Button(root, text="Start Training", command=start_training)
start_button.pack()

root.mainloop()
