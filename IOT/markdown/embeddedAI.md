# Embedded AI systems

## Abstract

Embedded artificial intelligence (AI) stands at the forefront of technological innovation, presenting novel opportunities for cost-effective and energy-efficient AI solutions that extend beyond the capabilities of cloud-based AI technologies. This article explores the rapidly growing field of embedded AI, particularly its exponential growth within the chip market dedicated to edge AI. By 2024, an estimated 1.5 billion edge AI chips are anticipated to be sold, indicating a substantial surge in demand and deployment.

## Introduction

The evolution of embedded AI introduces a paradigm shift, necessitating expertise that transcends traditional embedded systems, data science, and machine learning (ML). This technology demands a practical understanding of devices, sensors, and advanced real-time signal processing techniques. Whether dealing with video, audio, motion, or other signals, embedded AI's distinctive characteristics require a unique skill set for effective implementation.

## Embedded AI Landscape

At its core, embedded AI is deeply intertwined with sensors and data. The transformative potential of embedded AI lies in the adept extraction of meaningful information from diverse datasets. This article delves into the pivotal role of understanding data extraction techniques and explores how the predictive prowess of machine learning algorithms can be harnessed to process this data. The focus is on addressing a spectrum of real-world challenges through the application of embedded AI methodologies.


## What Can We Learn From Data?

For the purpose of illustration we will use the [NASA Turbofan Jet Engine Data Set](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps). This data set is the Kaggle version of the very well known public data set for asset degradation modeling from NASA. It includes Run-to-Failure simulated data from turbo fan jet engines.

The data set has a number of operational settings and data from 26 sensors. We show an extract of this below in table 1. 

| unit number | time | in cycles | setting 1 | setting 2 | setting 3 | sensor 1 |  sensor 2 | .... | sensor 26 |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- |  ----- | ----- | ----- |
| 1| 1| -0.0007| -0.0004| 100.0| 518.67| 641.82| 1589.70| | 23.4190| | 
| 1| 2| 0.0019| -0.0003| 100.0| 518.67| 642.15| 1591.82| | 23.4236| | 
| 1| 3| -0.0043| 0.0003| 100.0| 518.67| 642.35| 1587.99| | 23.3442| | 
| 1| 4| 0.0007| 0.0000| 100.0| 518.67| 642.35| 1582.79| | 23.3739| | 
| 1| 5| -0.0019| -0.0002| 100.0| 518.67| 642.37| 1582.85| | 23.4044| | 
| .....| .....| .....| .....| .....| ......| ......| ......| .....| 
| 1| 7| 0.0010| 0.0001| 100.0| 518.67| 642.48| 1592.32| | 23.3774| 

*Table 1: Extract from NASA Turbofan Jet Engine Data Set.*



Out first task is to read in the dataset into a format we can work with. The original data exists as a sequence of space-separated columns. We will create a Pandas frame to hold the data.

```python

import pandas as pd

dir_path = './data/' # identify the directory path that holds the data

                 
# define column names for easy indexing
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['s_{}'.format(i) for i in range(1,22)] # this will label the sensor data as s_1, s_2 etc
col_names = index_names + setting_names + sensor_names # combine these into a single reference for the column names

# read data
train = pd.read_csv((dir_path+'train_FD001.txt'), sep='\s+', header=None, names=col_names)

```

Having imported the data we can then examine the data for sensor number two.

```txt
0        641.82
1        642.15
2        642.35
3        642.35
4        642.37
          ...  
20626    643.49
20627    643.54
20628    643.42
20629    643.23
```


