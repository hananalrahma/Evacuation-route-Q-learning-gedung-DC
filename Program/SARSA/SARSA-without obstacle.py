import tkinter as tk
import random
import matplotlib.pyplot as plt
import threading
import time

# Grid size
grid_width = 50
grid_height = 20

# Initialize grid environment with obstacles
environment = [[0] * grid_width for _ in range(grid_height)]

# Start and goal positions
start_position = (0, 0)
goal_position = (grid_height - 1, grid_width - 1)

# Function to set obstacles in the environment
def set_obstacle(row, col):
    environment[row][col] = 1

# Function to set the start position of the agent
def set_start_position(row, col):
    global start_position
    start_position = (row, col)
    draw_agent_and_goal()

# Function to set the goal position of the agent
def set_goal_position(row, col):
    global goal_position
    goal_position = (row, col)
    draw_agent_and_goal()

# Initialize Q-table
q_table = [[0] * 4 for _ in range(grid_width * grid_height)]

# Initialize Q-table values
for row in range(grid_height):
    for col in range(grid_width):
        state = row * grid_width + col
        for action in range(4):
            q_table[state][action] = 0

# Function to choose action using epsilon-greedy policy
def get_action(state):
    epsilon = 0.1
    if random.random() < epsilon:
        return random.randint(0, 3)
    else:
        return q_table[state].index(max(q_table[state]))
       
# Function to get reward based on the environment
def get_reward(state, environment):
    row = state // grid_width
    col = state % grid_width
    if environment[row][col] == 1:
        return -10  # If obstacle
    elif row == goal_position[0] and col == goal_position[1]:
        return 10  # If goal is reached
    else:
        return -1  # Normal step

# Function to update Q-table based on SARSA algorithm
def update_q_table(state, action, next_state, next_action, reward):
    alpha = 0.1  # Learning rate
    gamma = 0.9  # Discount factor
    q_table[state][action] += alpha * (reward + gamma * q_table[next_state][next_action] - q_table[state][action])

# Function to find the shortest path using the Q-table
def find_shortest_path():
    shortest_path = []
    state = start_position[0] * grid_width + start_position[1]
    shortest_path.append(state)

    while state != goal_position[0] * grid_width + goal_position[1]:
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

# Function to draw the environment grid
def draw_environment(environment):
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
    # root.after(10, root.update)

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
    # root.after(10, root.update)

def check_convergence(episode_rewards):
    # Mengecek konvergensi berdasarkan perbedaan rewards episode sebelumnya dan saat ini
    if len(episode_rewards) > 1:
        last_reward = episode_rewards[-2]
        current_reward = episode_rewards[-1]
        diff = abs(current_reward - last_reward)
        if diff < 0.1:
            return True
    return False

# Function to train SARSA and find the shortest path
def train_sarsa():
    num_episodes = 3000
    episode_rewards = []
    episode_steps = []
    converged_episode = -1
    
    start_time = time.process_time()

    for episode in range(num_episodes):
        state = start_position[0] * grid_width + start_position[1]
        done = False
        episode_reward = 0
        episode_step = 0

        action = get_action(state)

        while not done:
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
            next_action = get_action(next_state)
            reward = get_reward(next_state, environment)

            update_q_table(state, action, next_state, next_action, reward)

            state = next_state
            action = next_action

            episode_reward += reward
            episode_step += 1

            if row == goal_position[0] and col == goal_position[1]:
                done = True

        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)
        
        if check_convergence(episode_rewards) and converged_episode == -1:
            converged_episode = episode
            end_time = time.process_time()
            print("Time to Convergence:", end_time - start_time, "seconds")

            # time.sleep(0.001)
            
    shortest_path = find_shortest_path()
    print("Shortest Path:")
    print(shortest_path)
    
    shortest_path_length = display_shortest_path_length()
    print("Shortest Path Length:", shortest_path_length)

    print("Converged Episode:", converged_episode)
    
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
        # root.after(10, root.update)

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
    training_thread = threading.Thread(target=train_sarsa)
    training_thread.start()
    

# Function to initialize obstacles

obstacles = [
    # List of obstacle coordinates
    (0, 3), (1, 3), (2, 3), (3, 3), (3, 2),
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
    (15, 32), (15, 33), (15, 34),
    (15, 35), (16, 35), (17, 35), (18, 35),
    (10, 12), (10, 13), (10, 14), (10, 15), (10, 16),
    (10, 17), (10, 18), (10, 19), (10, 21),
    (10, 20), (9, 20), (8, 20), (7, 20), (6, 20), (10, 46),
    (5, 20), (4, 20), (3, 20), (2, 20), (1, 20), (0, 20),
    (10, 22), (10, 23), (10, 24), (10, 25), (10, 26),
    (10, 27), (10, 28), (10, 34), (10, 33),
    (10, 34), (10, 35), (10, 36), (10, 37), (10, 38),
    (10, 39), (10, 40), (9, 46), (10, 45), (10, 41),
    (8, 46), (7, 46), (6, 46), (5, 46), (4, 46), (3, 46),
    (2, 46), (1, 46), (0, 46), (9, 33), (8, 33), (7, 33),
    (6, 33), (5, 33), (4, 33), (3, 33), (2, 33), (1, 33),
    (0, 33), (10, 32), (19, 5), (19, 15), (19, 28), 
    (19, 35), (10, 20)
    # (10, 29), (10, 33), (10, 41), (10, 42), (10, 46), (15, 31), (15, 30)
]

for obstacle in obstacles:
    row, col = obstacle
    if row < grid_height and col < grid_width:
        set_obstacle(row, col)


draw_environment(environment)
draw_agent_and_goal()

# Mengatur posisi awal agen
set_start_position(0, 0)


# Mengatur posisi tujuan agen
set_goal_position(grid_height - 8, grid_width - 50)

# Button to start SARSA training
start_button = tk.Button(root, text="Start SARSA Training", command=start_training)
start_button.pack()


root.mainloop()
