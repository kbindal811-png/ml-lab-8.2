
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load maze dataset
maze_df = pd.read_csv("maze.csv", header=None, dtype=str)
maze = maze_df.values.tolist()

rows = len(maze)
cols = len(maze[0])

start = None
goal = None

for r in range(rows):
    for c in range(cols):
        if maze[r][c] == "S":
            start = (r, c)
        elif maze[r][c] == "G":
            goal = (r, c)

if start is None or goal is None:
    raise ValueError("Maze must contain S and G.")

# Define possible actions
actions = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]

action_names = ["Up", "Down", "Left", "Right"]

# Create Q-table
q_table = np.zeros((rows, cols, len(actions)))

# Learning parameters
learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995

episodes = 5000
max_steps = rows * cols * 4

# Check valid movement
def is_valid(r, c):
    return (
        0 <= r < rows
        and 0 <= c < cols
        and maze[r][c] != "1"
    )

# Train Q-Learning agent
episode_rewards = []

for episode in range(episodes):

    state = start
    total_reward = 0

    for step in range(max_steps):

        r, c = state

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = np.random.randint(4)
        else:
            action = np.argmax(q_table[r, c])

        dr, dc = actions[action]
        nr, nc = r + dr, c + dc

        if not is_valid(nr, nc):
            next_state = state
            reward = -10
            done = False

        elif (nr, nc) == goal:
            next_state = (nr, nc)
            reward = 100
            done = True

        else:
            next_state = (nr, nc)
            reward = -1
            done = False

        # Update Q-value
        old_value = q_table[r, c, action]

        if done:
            next_max = 0
        else:
            next_max = np.max(
                q_table[next_state[0], next_state[1]]
            )

        q_table[r, c, action] = old_value + learning_rate * (
            reward + discount_factor * next_max - old_value
        )

        state = next_state
        total_reward += reward

        if done:
            break

    episode_rewards.append(total_reward)

    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )

# Save Q-table
q_rows = []

for r in range(rows):
    for c in range(cols):
        q_rows.append(
            [r, c] + q_table[r, c].tolist()
        )

pd.DataFrame(
    q_rows,
    columns=["row", "column"] + action_names
).to_csv("q_table.csv", index=False)

# Find learned path
state = start
learned_path = [state]
visited = {state}

for step in range(max_steps):

    r, c = state
    ranked_actions = np.argsort(q_table[r, c])[::-1]

    next_state = None

    for action in ranked_actions:
        dr, dc = actions[action]
        nr, nc = r + dr, c + dc

        if is_valid(nr, nc) and (nr, nc) not in visited:
            next_state = (nr, nc)
            break

    if next_state is None:
        break

    state = next_state
    learned_path.append(state)
    visited.add(state)

    if state == goal:
        break

# Display results
if learned_path[-1] == goal:
    print("Goal reached successfully!")
else:
    print("Goal not reached. Increase training episodes.")

print("Learned Path:", learned_path)
print("Total Moves:", len(learned_path) - 1)

# Save learned path
with open("learned_path.txt", "w") as file:
    file.write("Learned Path:\n")
    file.write(str(learned_path))
    file.write("\nTotal Moves: ")
    file.write(str(len(learned_path) - 1))

# Visualize maze and path
display_maze = np.zeros((rows, cols))

for r in range(rows):
    for c in range(cols):
        if maze[r][c] == "1":
            display_maze[r, c] = 1

plt.figure(figsize=(7, 7))
plt.imshow(display_maze, cmap="gray_r")

path_rows = [p[0] for p in learned_path]
path_cols = [p[1] for p in learned_path]

plt.plot(
    path_cols,
    path_rows,
    marker="o",
    linewidth=2,
    label="Learned Path"
)

plt.scatter(start[1], start[0], s=150, marker="s", label="Start")
plt.scatter(goal[1], goal[0], s=150, marker="*", label="Goal")

for r in range(rows):
    for c in range(cols):
        if maze[r][c] == "1":
            plt.text(c, r, "X", ha="center", va="center")
        elif maze[r][c] == "S":
            plt.text(c, r, "S", ha="center", va="center")
        elif maze[r][c] == "G":
            plt.text(c, r, "G", ha="center", va="center")

plt.xticks(range(cols))
plt.yticks(range(rows))
plt.grid()
plt.title("Q-Learning Maze Solver")
plt.legend()
plt.tight_layout()
plt.savefig("maze_solution.png")
plt.show()

# Plot training rewards
plt.figure(figsize=(8, 4))
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning Training Rewards")
plt.grid()
plt.tight_layout()
plt.savefig("training_rewards.png")
plt.show()
