# class-scheduling  
### Regression-informed local search for finding valid and optimal class schedules  
Input: 1) courses of interest, 2) schedules generated by Schedule-Builder, a valid-class schedule generator we have at UMN that has a limit of 15 schedules, and my ratings  

Methods:
1) designed metrics to fit my preferences such as the time of the earliest class in the schedule, number of classes in a day, desired credit amount, etc.
2) regression on metrics to get an objective function (instead of arbitrarily picking and tuning an objective function)  
3) local search (first-choice hill climbing and simulated annealing) to get good schedules based on the objective function heuristic  

final project for Artificial Intelligence 1 @ UMN Fall '23 taught by Andy Exley  
