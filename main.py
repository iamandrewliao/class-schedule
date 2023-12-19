import pandas as pd
import copy
import random
import re
import math
import numpy as np
from datetime import datetime
import itertools
from sklearn import linear_model
import statsmodels.api as sm

desired_credits = int(input("Enter your desired credit amount: "))
# desired_credits = 10
iterations = int(input("Number of iterations (at least a few hundred suggested): "))

data = pd.read_csv("classes.csv")
# this dataframe represents all the possible courses I can choose from & their respective data
# this allows our schedule to be represented simply since we can always reference class info in this dataframe

# convert something like "Mo, We, Fri" into ["Mo", "We", "Fr"]
def days_to_array(days):
    separated_days = re.findall(r"[A-Z][a-z]?", days)
    return separated_days

data.class_days = data.class_days.apply(days_to_array)

# converts something like "2:15 PM" to 1425
def time_str_to_int(time_str):
    # this converts something like "2:15 PM" to "14:15"
    time_str = datetime.strptime(time_str, "%I:%M %p").time().strftime("%H:%M")
    time_components = time_str.split(":")  # getting ["14", "15"] from "14:15"
    hours = int(time_components[0])
    minutes = int(time_components[1])
    # convert to the desired integer representation (60 minute scale -> out of 100)
    time_integer = hours * 100 + int((minutes / 60) * 100)
    return time_integer

# print(time_str_to_int("2:15 PM"))

data.class_time_start = data.class_time_start.apply(time_str_to_int)
data.class_time_end = data.class_time_end.apply(time_str_to_int)
# print(type(data.class_time_start[0]))

# defining this for later use
possible_days = ["Mo", "Tu", "We", "Th", "Fr"]

# We start with a partial assignment:
# represent a schedule (a state):
start_schedule = {"Mo": ["CSCI8980-002"],
            "Tu": ["EE8231", "PSY8036"],
            "We": ["ME5248", "CSCI8980-002"],
            "Th": ["EE8231"],
            "Fr": ["ME5248"]}

# for testing purposes
overlap_schedule = {"Mo": ["MATH5466"],
            "Tu": ["ME3222"],
            "We": ["MATH5466"],
            "Th": ["ME3222", "BMEN5151"],
            "Fr": ["ME5248"]}

# for testing purposes
nan_schedule = {"Mo": ["CSCI8980-001", "CSCI8980-003"],
            "Tu": ["CSCI5541"],
            "We": ["CSCI8980-001", "CSCI8980-003"],
            "Th": ["CSCI5541"],
            "Fr": []}

def classes_in_schedule(schedule):  # helper function for some metric functions
    classes = []
    for day in schedule:
        for cls in schedule[day]:
            if cls not in classes:
                classes.append(cls)
    return classes


# METRICS
def count_creds(schedule):
    classes = classes_in_schedule(schedule)
    creds = 0
    for cls in classes:
        creds += data[data.classes == cls].creds.values[0]
    return creds

def credit_limit_metric(schedule, desired_creds):  # penalize schedules that have a total credit amt != our desired amt
    creds = count_creds(schedule)
    metric = -1*(creds-desired_creds)**2  # penalizes exponentially ~ -x^2
    return metric

def overlap_metric(schedule):  # penalizes schedules that overlap
    for day, classes in schedule.items():
        if len(classes) > 1:  # we can ignore days with just one class or less
            for class_pair in itertools.combinations(classes, 2):  # look at all the combinations
                class1, class2 = class_pair
                # get start and end times for each class
                start_time1 = data[data.classes == class1].class_time_start.values[0]
                end_time1 = data[data.classes == class1].class_time_end.values[0]
                start_time2 = data[data.classes == class2].class_time_start.values[0]
                end_time2 = data[data.classes == class2].class_time_end.values[0]
                if ((start_time1<=end_time2 and start_time1>=start_time2) or
                        (start_time2<=end_time1 and start_time2>=start_time1)):
                    return -1
    return 0

# print(overlap_metric(overlap_schedule))  # should return -1 b/c there is an overlap

def early_metric(schedule):  # I don't prefer classes that start earlier than ~9:15/9:30
    classes = classes_in_schedule(schedule)
    metric = 0
    for cls in classes:
        # Using desmos, I wrote a function like -e^(-x) that will start penalizing if earlier than ~10am: -e^(-2x+18)
        # dividing class_time_start by 100 turns something like 1000 (10:00 AM) into 10 which works with this function
        metric += -math.pow(math.e, -2 * (data[data.classes == cls].class_time_start.values[0] / 100) + 18)
    return metric

# print(early_metric(test_schedule))

def num_classes_daily_metric(schedule):  # I don't want too many classes in a day
    # Find the day with the most classes in it and compute metric
    # I wrote a function like -e^x that will penalize a little for 2 classes, a lot more for 3 classes, etc.
    most_classes = 0
    for day in schedule:
        num_classes = len(schedule[day])
        if num_classes > most_classes:
            most_classes = num_classes
    return -math.pow(math.e, most_classes - 3)

def avg_interest_lvl_metric(schedule):  # average interest_lvl across classes
    classes = classes_in_schedule(schedule)
    ratings = []
    for cls in classes:
        ratings.append(data[data.classes == cls].interest_lvl.values[0])
    return np.nanmean(ratings)  # to get the average

def avg_ratemyprof_metric(schedule):  # average ratemyprof reviews across classes
    classes = classes_in_schedule(schedule)
    ratings = []
    for cls in classes:
        ratings.append(data[data.classes == cls].ratemyprof.values[0])
    return np.nanmean(ratings)  # to get the average

# print(avg_ratemyprof_metric(start_schedule))

def avg_avg_grade_metric(schedule):  # average avg_grade across classes
    classes = classes_in_schedule(schedule)
    ratings = []
    for cls in classes:
        ratings.append(data[data.classes == cls].avg_grade.values[0])
    return np.nanmean(ratings)  # to get the average


# # MULTIPLE REGRESSION USING METRICS TO FIND OBJECTIVE FUNCTION COEFFICIENTS
# # STEP 1: PREPARE DATA
# schedules_data = pd.read_csv("schedules.csv")  # each row is a schedule and the corresponding rating
# schedules_data.fillna('', inplace=True)
#
# # make a dataset where features are metrics of the schedules and labels are ratings
# schedules_only_data = schedules_data.loc[:, schedules_data.columns != "rating"]
#
# # I have to convert each row to a schedule like this:
# # overlap_schedule = {"Mo": ["MATH5466"],
# #             "Tu": ["ME3222"],
# #             "We": ["MATH5466"],
# #             "Th": ["ME3222", "BMEN5151"],
# #             "Fr": ["ME5248"]}
#
# idx_to_day = {0: "Mo", 1: "Tu", 2: "We", 3: "Th", 4: "Fr"}
# metrics = []
# for row in schedules_only_data.to_numpy():
#     schedule_metrics = {}
#     cur_schedule = {}
#     iterate_schedule = 0
#     for day_schedule in row:
#         if day_schedule == '':
#             cur_schedule[idx_to_day[iterate_schedule]] = []
#         elif ',' in day_schedule:  # if multiple classes in a day
#             cur_schedule[idx_to_day[iterate_schedule]] = day_schedule.split(", ")
#         else:
#             cur_schedule[idx_to_day[iterate_schedule]] = [day_schedule]
#         iterate_schedule += 1
#     schedule_metrics["credit_limit"] = credit_limit_metric(cur_schedule, desired_credits)
#     schedule_metrics["overlap"] = overlap_metric(cur_schedule)
#     schedule_metrics["early"] = early_metric(cur_schedule)
#     schedule_metrics["num_classes_daily"] = num_classes_daily_metric(cur_schedule)
#     schedule_metrics["avg_interest_lvl"] = avg_interest_lvl_metric(cur_schedule)
#     schedule_metrics["avg_ratemyprof"] = avg_ratemyprof_metric(cur_schedule)
#     schedule_metrics["avg_avg_grade"] = avg_avg_grade_metric(cur_schedule)
#     metrics.append(schedule_metrics)
#
# metrics_df = pd.DataFrame(metrics)
# # print(metrics_df.assign(rating=schedules_data.rating))
#
# # STEP 2: RUN MULTIPLE REGRESSION
# metrics_df_with_rating = metrics_df.assign(rating=schedules_data.rating)
#
# X = metrics_df_with_rating.loc[:, metrics_df_with_rating.columns != "rating"]
# y = metrics_df_with_rating['rating']
#
# # with sklearn
# regr = linear_model.LinearRegression()
# regr.fit(X, y)
#
# print('Intercept: \n', regr.intercept_)
# print('Coefficients: \n', regr.coef_)
#
# # with statsmodels
# X = sm.add_constant(X)  # adding a constant
#
# model = sm.OLS(y, X).fit()
# predictions = model.predict(X)
#
# print_model = model.summary()
# print(print_model)

# The coefficients we find from multiple regression are:
regression_coeffs = [10.2731, 0.2370, 6.0287, 4.718e-06, 0.2598, 2.1757, 0.0245, -6.0618]
# corresponding to the intercept and the metrics:
# [intercept, credit_limit, overlap, early, num_classes_daily, avg_interest_lvl, avg_ratemyprof, avg_avg_grade]
# Our adjusted R-squared is good!

def obj_func(schedule, regression_coefficients):  # construct the obj_func from our regression results
    score = 0
    intercept = regression_coefficients[0]
    score += intercept
    # print(score)
    metric_coeffs = regression_coefficients[1:]
    metrics_array = [credit_limit_metric(schedule, desired_credits),
                     overlap_metric(schedule),
                     early_metric(schedule),
                     num_classes_daily_metric(schedule),
                     avg_interest_lvl_metric(schedule),
                     avg_ratemyprof_metric(schedule),
                     avg_avg_grade_metric(schedule)]
    # print(metrics_array)
    for (coeff, met) in zip(metric_coeffs, metrics_array):
        # print(coeff, met)
        if not np.isnan(met):  # metric can be nan if all its values are nan
            score += coeff*met
    return score

# good_schedule = {"Mo": ["MATH5466", "CSCI8980-003"],
#             "Tu": ["PSY8036"],
#             "We": ["MATH5466", "CSCI8980-003"],
#             "Th": [],
#             "Fr": []}
#
# awful_schedule = {'Mo': ['CSCI8980-002', 'CSCI8980-003', 'MATH5466', 'CSCI8980-001'],
#                   'Tu': [],
#                   'We': ['CSCI8980-002', 'CSCI8980-003', 'MATH5466', 'CSCI8980-001', 'EE8215', 'ME5248'],
#                   'Th': [],
#                   'Fr': ['EE8215', 'ME5248']}
#
# print(obj_func(good_schedule, regression_coeffs))
# print(obj_func(awful_schedule, regression_coeffs))

def add_class(schedule, classes):  # add course(s) to schedule; helper function for generate_schedule()
    for i in classes:  # for each class I want to add in the schedule
        for j in data[data.classes == i].class_days.values[0]:  # for each day that the class meets
            schedule[j].append(i)
    return schedule

def classes_not_on_schedule(schedule):  # helper function for generate_schedule()
    scheduled_classes = set(class_ for classes in schedule.values() for class_ in classes)
    classes = set(data.classes.tolist()) - scheduled_classes
    return list(classes)

# print(classes_not_on_schedule(start_schedule))

def generate_schedule(schedule):
    '''
    process for generating a new schedule:
    choose a random day and swap a class out with either 0, 1, or 2 classes that aren't assigned
    I do this to vary the # classes on the schedule, which is necessary b/c not all classes are the same # credits
    '''
    # make copy to edit; this will be the new or "child" schedule
    schedule = copy.deepcopy(schedule)
    # we will swap in a random number of classes (between 0 and 2 to keep it reasonable) for each class we take out
    random_swap_num = random.randint(0,2)
    # the classes we choose to swap in need to not be in our schedule
    random_swap_classes = random.sample(classes_not_on_schedule(schedule), random_swap_num)
    # print(f"swap_num: {random_swap_num}, swap_in_classes: {random_swap_classes}")
    # choose random day
    random_day = random.choice(possible_days)
    if len(schedule[random_day]) > 0:  # if that day isn't blank
        # choose a random class from that day to account for days w/ multiple classes
        random_class = random.choice(schedule[random_day])
        # print(f"random_day: {random_day}, swap_out_class: {random_class}")
        # need to make sure to remove the class in its entirety (e.g. if a class meets on more than one day)
        # so we go through all the days the class meets
        for i in data[data.classes == random_class].class_days.values[0]:
            schedule[i].remove(random_class)
    add_class(schedule, random_swap_classes)
    # print(schedule)
    return schedule

# LOCAL SEARCH ALGORITHMS
# HILL CLIMBING
def first_choice_hill_climbing(state, max_iter):
    current_state = state
    current_score = obj_func(state, regression_coeffs)
    for i in range(1, max_iter+1):
        schedule = generate_schedule(current_state)
        score = obj_func(schedule, regression_coeffs)
        if score > current_score:
            current_state = schedule
            current_score = score
    return current_state, current_score

schedule_hillclimb, score_hillclimb = first_choice_hill_climbing(start_schedule, iterations)
# print(score_hillclimb)
# print("score components: ", [credit_limit_metric(schedule_hillclimb, desired_credits),
#                              overlap_metric(schedule_hillclimb),
#                              early_metric(schedule_hillclimb),
#                              num_classes_daily_metric(schedule_hillclimb),
#                              avg_interest_lvl_metric(schedule_hillclimb),
#                              avg_ratemyprof_metric(schedule_hillclimb),
#                              avg_avg_grade_metric(schedule_hillclimb)])
# print(schedule_hillclimb)

# SIMULATED ANNEALING
def exponential_cooling(initial_temperature, cooling_rate, iteration):
    temperature = initial_temperature
    for i in range(iteration):
        temperature *= cooling_rate
    return temperature

def linear_cooling(initial_temperature, cooling_rate, iteration):
    temperature = initial_temperature
    for i in range(1, iteration+1):
        temperature -= cooling_rate * i
    return temperature

def logarithmic_cooling(initial_temperature, iteration):
    temperature = initial_temperature
    for i in range(iteration):
        temperature = 100.0 / math.log(3 + i)
    return temperature

def simulated_annealing(state, cooling_schedule, max_iter):
    initial_temp = 100.0
    current_state = state
    current_score = obj_func(state, regression_coeffs)
    best_state = current_state
    best_score = current_score
    for t in range(1, max_iter+1):
        if cooling_schedule == "logarithmic":
            T = logarithmic_cooling(initial_temp, t)
        elif cooling_schedule == "exponential":
            T = exponential_cooling(initial_temp, 0.99, t)
        elif cooling_schedule == "linear":
            T = linear_cooling(initial_temp, 1, t)
        else:
            return "Not a valid schedule, choose from ('logarithmic', 'exponential', 'linear')"
        if T == 0:
            print('Temperature reached 0')
            return current_state, current_score
        successor = generate_schedule(current_state)
        delta = (obj_func(successor, regression_coeffs)
                 - obj_func(current_state, regression_coeffs))
        # print(f"delta: {delta} | temperature: {T}")
        if delta > 0 or random.random() < math.exp(delta/T):
            current_state = successor
            current_score = obj_func(successor, regression_coeffs)
        if current_score > best_score:
            best_score = current_score
            best_state = current_state
    return best_state, best_score

schedule_simann, score_simann = simulated_annealing(start_schedule, 'logarithmic', iterations)

scores = [score_hillclimb, score_simann]

print("Best Schedule Found:\n")
if max(scores) == score_hillclimb:
    print("Algorithm: First Choice Hill Climbing\n")
    print(f"After {iterations} iterations\n"
          f"Schedule found with First-Choice Hill Climbing: {schedule_hillclimb}\n"
          f"Score: {score_hillclimb}\n"
          f"Creds: {count_creds(schedule_hillclimb)}")
else:
    print("Algorithm: Simulated Annealing\n")
    print(f"After {iterations} iterations\n"
          f"Schedule found with Simulated Annealing: {schedule_simann}\n"
          f"Score: {score_simann}\n"
          f"Creds: {count_creds(schedule_simann)}")

