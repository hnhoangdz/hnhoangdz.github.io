import numpy as np
import matplotlib.pyplot as plt

# Entropy
def _entropy(p):
  return -(p*np.log2(p)+(1-p)*np.log2((1-p)))

# Probability p
p = np.linspace(0.001, 0.999, 200)

# Visualize
entropy = _entropy(p)
plt.figure(figsize = (12, 8))
plt.plot(p, entropy)
plt.xlabel('p')
plt.ylabel('entropy')
plt.title('Entropy')
plt.show()

# print(np.log2(2))