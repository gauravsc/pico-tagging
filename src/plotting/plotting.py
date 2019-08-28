import matplotlib.pyplot as plt

# input values for drawing the plot
noise = [0.0, 0.3, 1.0, 2.0]
microf1 = [0.601, 0.605, 0.608, 0.603]

# draw the plot
fig, ax = plt.subplots()
ax.plot(noise, microf1, marker='o')
ax.legend()
ax.set_ylabel("MicroF1")

# save the plot in a figure
plt.savefig('../plots/microf1_and_noise.png')



# input values for drawing the plot
noise = [0.0, 0.3, 1.0, 2.0]
macrof1 = [0.044, 0.0437, 0.045, 0.0429]

# draw the plot
fig, ax = plt.subplots()
ax.plot(noise, macrof1, marker='*')
ax.legend()
ax.set_ylabel("MacroF1")

# save the plot in a figure
plt.savefig('../plots/macrof1_and_noise.png')

