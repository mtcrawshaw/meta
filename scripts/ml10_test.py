import numpy as np

from meta.train.ml_benchmarks import ML10, ML45


env = ML45.get_test_tasks()

print("num tasks: %s" % env.num_tasks)

"""
for i in range(env.num_tasks):

    # Sample new task and reset environment.
    tasks = env.sample_tasks(1)
    env.set_task(tasks[0])
    obs = env.reset()  # Reset environment

    # Take a step.
    # print("%d initial obs: %s" % (i, str(obs)))
    print(str(obs))
    a = env.action_space.sample()
    obs, reward, done, info = env.step(a)

    # print("%d second obs: %s\n" % (i, str(obs)))
    print(str(obs))
    print("\n")
"""

tasks = []
for _ in range(1000):
    env.set_task(env.sample_tasks(1)[0])
    task = env.active_task
    if task not in tasks:
        tasks.append(task)

tasks = sorted(tasks)
print(tasks)
