import matplotlib.pyplot as plt
import numpy as np

smoothing = 10

values = []
with open('./multimodal', 'r') as f:
    buffer = []
    step = 0
    for line in f:
        l = line.split(" : ")[1]
        l = l.split(",")[1]
        score = float(l.split()[2])
        step += 1
        if len(buffer) == smoothing:
            buffer = buffer[1:]
            buffer.append(score)
            values.append(sum(buffer) / smoothing)
        else:
            buffer.append(score)

plt.title("Learning curve by averaging last {} episodes".format(smoothing))
plt.xlabel("Step", fontweight='bold')
plt.ylabel('Score of the agent on the game', fontweight='bold')
plt.plot(values)
plt.savefig("./multimodal.png")
