# Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation

## Data Description

Data Set: FD001
Description: Remaining Useful Life (RUL)
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

## Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine (identified by unit number) – i.e., the data can be considered to be from a fleet of engines of the same type. The are 100 different engines with each engine starting with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series.

In the training set, the fault grows in magnitude until system failure. So the maximum numner of cycles for a given engine or unit number is the failure point.

In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of 100 true Remaining Useful Life (RUL) values for the test data with an entry corresponding to each unit number (engine group).

The data is provided as two text files (one for test called test_FD001.txt and the second for training called train_FD001.txt). Each file has 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:

- unit number (numbered from 1 to 100)
- time, in cycles
- operational setting 1
- operational setting 2
- operational setting 3
- sensor measurement 1
- sensor measurement 2
- sensor measurement ... 26

The file RUL_FD001.txt contains one 100 rows of a single column with the rue Remaining Useful Life (RUL) values for the test data. Each row corresponds to a specifc unit number.

## References

- [CMAPSSData](https://github.com/mapr-demos/predictive-maintenance/tree/master/notebooks/jupyter/Dataset/CMAPSSData)
- A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
