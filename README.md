# Handling concept drift in time series datastream of a robot

## Project Overview
This project is about evaluating drift detection algorithm
in the context of detecting new terrain of a hexapod robot.
The dataset consists of six types of terrains : Black Flat, Blacks Rough, Flat, Wooden Cube, Black Rough, Grass Rough 
## Content of the repository
### ADWIN 
ADWIN is a drift detection algorithm with the based principle of shrinking the window whever a change occurs. For more
information, check 'Learning from time-changing data with Adaptive Window' from Albert Bifet.

### Page-Hinkey Test
P-H test is a drift detection algorithm design for gradual changes.
The code has been inspired from : "Knowledge Discovery from Data Streams."
### Kolmogorov 
We have done a light adaptation of the K-s test to the context of Drift detection.
For a more detailed description of this work. See the master's 1 thesis link to it [Presentation slides](https://docs.google.com/presentation/d/1sHhPBRP0vLswpTIy46BaCoLnSdbp6TqDp0HQrs9K8p4/edit?usp=sharing)
