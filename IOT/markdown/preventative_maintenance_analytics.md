# Data Analytics


## Measurement

### Variance

The variance is a squared measure of how much each individual process temperature value deviates from the mean.The larger the variance, the more spread out the data points are from the mean. The unit of variance is squared, so it doesn't have the same unit as the original data. If you want a measure in the same unit as your original data, you can take the square root of the variance to get the standard deviation.

\[ \text{variance} = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1} \]

### Standard Deviation

The standard deviation is a statistical measure of the amount of variation or dispersion in a set of values. It quantifies how much individual data points differ from the mean (average) of the data set. A low standard deviation indicates that the data points tend to be close to the mean, while a high standard deviation indicates that the data points are spread out over a wider range.


\[ \sigma = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}} \]



### Z-Score

A Z-score is a statistical measure that describes a value's relationship to the mean of a group of values. It is measured in terms of standard deviations from the mean. A Z-score of 0 indicates that the data point's score is identical to the mean score, a Z-score of 1.0 indicates a value that is one standard deviation from the mean, and so on. The Z-Score can be useful in determining data spikes.

\[ Z = \frac{(X - \mu)}{\sigma} \]

### Min/Max

The Minimum and maximum values can emphasize outliers. This can be useful in determining anomalies in a dataset.


### Tumbling or Rolling Window

Tumbling window functions group data streams into time segments. Tumbling windows means that the window does not repeat or overlap data from one segment waterfall into the next.

### Hopping Windows

Hopping windows are tumbling windows that overlap. They allow you to set specific commands and conditions, such as every 5 minutes, give me the readings over the last 10 minutes. To make a hopping window the same as a tumbling window, you would make the hop size the same as the window size.

### Sliding Windows

There are two types of sliding windows. Time Sliding triggers at regular interval and Eviction Sliding triggers on a count.


## Using Charts

### Bar Charts

Bar charts are commonly used in various circumstances to visually represent and compare categorical data. Bar charts can have many items simply because of the page layout. Column and bar charts can also show change over time. Bar charts can show the distribution of data across different categories. This is useful when you want to see how data points are spread among different groups. They can also be used to represent data over time, especially when the time intervals are discrete (e.g., days, months). Each bar represents the value of the data at a specific time point. Bar charts are useful for displaying the frequency distribution of categorical data. For example, the number of occurrences of each category in a dataset.

### Scatter Plot

A scatter plot is a type of data visualization that displays individual data points on a two-dimensional graph. Each point on the plot represents the values of two variables, with one variable plotted on the x-axis and the other on the y-axis. Scatter plots are useful for visualizing the relationship between two continuous variables, identifying patterns, and detecting outliers.

### Heatmap

A heatmap is a graphical representation of data where values in a matrix are represented as colors. It's a way of visualizing complex data in a two-dimensional space, and it is particularly useful for revealing patterns and trends.

### Co-variance

Covariance measures how much two variables change together. If the covariance is positive, it indicates that when one variable is above its mean, the other variable tends to be above its mean as well. If the covariance is negative, it indicates an inverse relationship, meaning that when one variable is above its mean, the other tends to be below its mean.

Covariance is sensitive to the scale of the variables, and its interpretation can be challenging when dealing with variables on different scales. To overcome this, the correlation coefficient is often used, as it standardizes the measure to be between -1 and 1, providing a normalized measure of the strength and direction of the linear relationship between two variables. The correlation coefficient is calculated as the covariance divided by the product of the standard deviations of the variables.

\[ \text{cov}(X, Y) = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{n-1} \]

A negative covariance indicates an inverse relationship: when one variable increases, the other tends to decrease, and vice versa. A negative covariance between temperature and rainfall suggests that, on average, when the air temperature is higher than its mean, there is a tendency for process temperature to be higher than its mean, and when the air temperature is lower than its mean, there is a tendency for process temperature  to be lower.

### Anomaly detection

Anomaly detection is a technique used in data analysis and machine learning to identify patterns, events, or observations that deviate significantly from the expected behavior in a dataset. These deviations are often referred to as "anomalies" or "outliers." Anomalies can represent unusual or suspicious activities, errors, or events that are different from the norm.

Anomaly detection in machine learning can be unsupervised, supervised, or semi-supervised. We can start by using an unsupervised machine learning algorithm (like K-means) to identify data clusters or patterns.

The data may represent various states of the system under examination. These states may be resting states, in-use states, cold states, or one of a number of failed states.


### Looking for evidence of clustering in the data

#### Tightness of Clusters

Look for clusters where data points within the same cluster are closely packed together. Tight and compact clusters are indicative of well-defined groups.

#### Separation Between Clusters

Check for clear separation between different clusters. Ideally, there should be noticeable gaps or boundaries between clusters, indicating that different groups are distinct from each other.

#### Centroid Locations

Examine the locations of cluster centroids. The centroids represent the mean or center of each cluster. Well-separated and distinct centroids contribute to evidence of meaningful clustering.

#### K Means

K-means is a popular and widely used clustering algorithm in machine learning and data analysis. It is an unsupervised learning algorithm designed to partition a dataset into K distinct, non-overlapping subsets (clusters), where each data point belongs to only one cluster. The algorithm seeks to minimize the sum of squared distances between data points and the centroid of their assigned cluster.


### Looking for Outliers in Data

Identifying outliers in our data is very important as they represent bad sensor placement, or a number of other issues. We can view outliers by looking at values that fall more than three standard deviations from the mean of our data.



## Strategies for Predictive Maintenance

### Using XGBoost for Prediction


XGBoost (eXtreme Gradient Boosting) is a powerful and popular machine learning algorithm that falls under the category of ensemble learning. It is particularly well-suited for regression and classification problems.

In this example we are adapting it to predict the need for preventative maintenance.

#### Mean Percentage Error (MPE)

MPE = (1/n) Σ ( (y_true - y_pred) / y_true ) * 100

In this formula:

- n is the number of samples.
- y_true s the true value.
- y_pred   is the predicted value.

#### Mean Squared Error (MSE)

MSE = (1/n) Σ (y_true - y_pred)^2