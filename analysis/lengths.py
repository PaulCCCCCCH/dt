import numpy as np
import matplotlib.pyplot as plt

ups = []
downs = []


for i in range(5):
    d = dict()
    l = dict()
    allow_eos_before_clipping = i
    with open("./out.txt", 'r') as f:

        for line in f:
            eos_count = 0
            action = int(line.split("]")[0][1])
            sentence = line.split("]")[1].strip()
            length = 0
            if action not in d.keys():
                d[action] = []
            if action not in l.keys():
                l[action] = 0
            d[action].append(sentence)
            count = 0
            for w in sentence.split():
                count += 1
                if w == "<eos>":
                    if eos_count == allow_eos_before_clipping:
                        break
                    eos_count += 1

            l[action] += count
    ups.append((l[2] + l[4]) / (len(d[2]) + len(d[4])))
    downs.append((l[3] + l[5]) / (len(d[3]) + len(d[5])))

print(ups)
print(downs)

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
plt.xlabel('End the sentence after seeing n EOS symbols', fontweight='bold')
plt.ylabel('Average sentence length', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(ups))], ['n=1', 'n=2', 'n=3', 'n=4', 'n=5'])

# Create legend & Show graphic
plt.legend()
plt.savefig("./lengths.png")
