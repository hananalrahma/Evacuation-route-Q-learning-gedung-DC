# import lilbrary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import fungsi dari env.py
from env import final_states

# membuat class untuk table Q-learning (Q-table)
class QLearningTable:
    def __init__(self, actions, Learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        # daftar actions
        self.actions = actions
        # learning rate
        self.lr = Learning_rate
        # nilai gamma
        self.gamma = reward_decay
        # nilai epsilon
        self.epsilon = e_greedy
        # membuat full Q-table untuk semua cells
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        # membuat Q-table untuk final route
        self.q_table_final = pd.DataFrame(columns=self.actions, dtype=np.float64)

    # fungsi untuk memilih aksi untuk agent
    def choose_action(self, observation):
        # mengecek jika ada state yang tersedia di dalam table
        self.check_state_exist(observation)
        # pemilihan tindakan 90%. berdasarkan epsilon == 0.9
        # memilih aksi terbaik
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # pemilihan tindakan acak yang tersisa 10% maka dipilihi aksi random
            action = np.random.choice(self.actions)
        return action
    
    # fungsi untuk learning dan update Q-table berdasarkan pengetahuan baru
    def learn(self, state, action, reward, next_state):
        # mengecek jika step selanjutnya tersedia di dalam Q-table
        self.check_state_exist(next_state)

        # state saat ni di posisi saat ini
        q_predict = self.q_table.loc[state, action]

        # mengecek apakah state selanjutnya free ataukah ada obstacle atau goal
        if next_state != 'goal' or next_state != 'obstacle':
            q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()
        else:
            q_target = reward
        
        # update Q-table dengan pengetahuan baru
        self.q_table.loc[state, action] += self.lr * (q_target-q_predict)

        return self.q_table.loc[state, action]
    
    # menambahkan state baru ke Q-table
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
    
    # print Q-table dengan state
    def print_q_tables(self):
        # mendapatkan koordinat final route dari env.py
        e = final_states()

        # mengkoparasikan imdex dengan koordinat dan menulis di dalam Q-table values
        for i in range(len(e)):
            state = str(e[i])
            # mengecek semua index
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of final Q-table =', len(self.q_table_final.index))
        print('Final Q-table with values from the final route:')
        print(self.q_table_final)

        print()
        print('Length of full Q-table=', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)
