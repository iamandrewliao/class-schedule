import pandas as pd
import copy
import random
import re

# desired_credits = int(input("Enter your desired credit amount: "))
# iterations = int(input("Number of iterations (1000+ suggested): "))

data = pd.read_csv("classes.csv")
# this dataframe represents all the possible courses I can choose from & their respective data
# this allows our schedule to be represented simply since we can always reference class info in this dataframe

# convert something like "Mo, We, Fri" into ["Mo", "We", "Fr"]
def days_to_array(days):
    separated_days = re.findall(r"[A-Z][a-z]?", days)
    return separated_days

data.class_days = data.class_days.apply(days_to_array)

# defining this for later use
possible_days = ["Mo", "Tu", "We", "Th", "Fr"]

# We start with a partial assignment:
# represent a schedule (a state):
start_schedule = {"Mo": ["CSCI8980-001", "CSCI8980-003"],
            "Tu": ["CSCI5541"],
            "We": ["CSCI8980-001", "CSCI8980-003"],
            "Th": ["CSCI5541"],
            "Fr": []}

def credit_limit_metric(schedule, desired_creds):
    '''
    we want to create a function that will penalize schedules that have a total credit amt != our desired amt
    '''
    classes = []
    for day in schedule:
        for cls in schedule[day]:
            if cls not in classes:
                classes.append(cls)
    creds = 0
    for cls in classes:
        creds += data[data.classes == cls].creds.values[0]
    metric = -1*(creds-desired_creds)**2  # penalizes exponentially
    return metric

# def overlap_metric(schedule):
#

# other metrics: class_start_time, avg interest_lvl, ratemyprof reviews, avg_grade,
# how close classes are to each other, # classes in a day

schedules_data = pd.read_csv("schedules.csv")
print(schedules_data)

# # def obj_func(schedule)
#

def add_class(schedule, classes):
    '''
    add course(s) to schedule; helper function for generate_schedule()
    '''
    for i in classes:  # for each class I want to add in the schedule
        for j in data[data.classes == i].class_days.values[0]:  # for each day that the class meets
            schedule[j].append(i)
    return schedule

def classes_not_on_schedule(schedule):
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

# generate_schedule(start_schedule)


