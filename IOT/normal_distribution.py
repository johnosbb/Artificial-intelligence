import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro


# Set the seed for reproducibility
np.random.seed(42)

# Generate a sample from a normal distribution with mean=0 and standard deviation=1
sample_size = 20000
mu, sigma = 0, 1
normal_distribution_sample = np.random.normal(mu, sigma, sample_size)

# Plot a histogram to visualize the distribution
plt.hist(normal_distribution_sample, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')

# Plot the probability density function (PDF) for comparison
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = np.exp(-(x - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))
plt.plot(x, p, 'k', linewidth=2)

plt.title('Sample from Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
test_statistic, p_value = shapiro(normal_distribution_sample)

print("Test Statistic:", test_statistic)
print("P-value:", p_value)