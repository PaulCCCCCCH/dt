
import numpy as np
import matplotlib.pyplot as plt

ups = []
downs = []

d = dict()
l = dict()
with open("./out.txt", 'r') as f:

    for line in f:
        eos_count = 0
        action = int(line.split("]")[0][1])
        sentence = line.split("]")[1].strip()
        length = 0
        if action not in d.keys():
            d[action] = []
        if action not in l.keys():
            l[action] = {}
            for i in range(10):
                l[action][i] = 0

        d[action].append(sentence)
        pos = 0
        for w in sentence.split():
            if w == "position":
                l[action][pos] += 1
                break

            pos += 1

ups = (np.array([l[2][pos] for pos in range(10)]) + np.array([l[4][pos] for pos in range(10)])) / (len(d[2]) + len(d[4]))
downs = (np.array([l[3][pos] for pos in range(10)]) + np.array([l[5][pos] for pos in range(10)])) / (len(d[3]) + len(d[5]))


# set width of bar
barWidth = 0.25

# Set position of bar on X axis
r1 = np.arange(len(ups))
r2 = [x + barWidth for x in r1]

# plt.title('Average instruction length by different clipping threshold')

# Make the plot
plt.bar(r1, ups, color='#7f6d5f', width=barWidth, edgecolor='white', label="'moving up' instructions")
plt.bar(r2, downs, color='#557f2d', width=barWidth, edgecolor='white', label="'moving down' instructions")

# Add xticks on the middle of the group bars
plt.xlabel("Word 'position' appears at position n of the sentence", fontweight='bold')
plt.ylabel('Frequency', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(ups))], ["n=" + str(i+1) for i in range(10)])

# Create legend & Show graphic
plt.legend()
plt.savefig("./positions.png")
