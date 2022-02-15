import matplotlib.pyplot as plt

losses = []

with open('losses_file.txt', 'r') as f:
    for line in f:
        losses.append(float(line))

print(losses)

plt.style.use(['science'])
plt.plot(losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss after given epoch')
plt.show()
