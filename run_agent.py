from env import Environment
from agent_brain import QLearningTable


def update():
    # Resulted list for the plotting Episodes via Steps
    steps = []

    # Summed costs for all episodes in resulted list
    all_costs = []

    for episode in range(5000):
        # observasi awal
        observation = env.reset()
        
        # update jumlah step untuk tiap episode
        i = 0

        # update cost untuk setiap episode
        cost = 0

        while True:
            # me-refresh env
            env.render()

            # RL memilih aksi berdasarkan observasi
            action = RL.choose_action(str(observation))

            # RL mengambil aksi dan mendapatkan observasi baru dan reward
            observation_, reward, done = env.step(action)

            # RL belajar dari transisi ini dan menghitung cost
            cost += RL.learn(str(observation), action, reward, str(observation_))

            # Swapping the observations - current and next
            observation = observation_

            # menghitung jumlah steps pada episode saat ini
            i += 1

            # break "while loop" jika sudah mencapai akhir dari episode saat ini
            # break ketika agen mencapai goal atau menabrak obstacle
            if done:
                steps += [i]
                all_costs += [cost]
                break

    # memperlihatkan final route
    env.final()

    # Showing the Q-table with values for each action
    RL.print_q_table()

    # Plotting the results
    RL.plot_results(steps, all_costs)


# Commands to be implemented after running this file
if __name__ == "__main__":
    # memanggil env
    env = Environment()
    # memanggil algoritma q-learning
    RL = QLearningTable(actions=list(range(env.n_actions)))
    # Running the main loop with Episodes by calling the function update().
    env.after(100, update)  # Or just update()
    env.mainloop()
