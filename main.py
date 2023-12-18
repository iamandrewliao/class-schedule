import pandas as pd
from ortools.sat.python import cp_model

data = pd.read_csv("classes.csv")
small_data = data.head(3)
classes = data.course.values.tolist()
small_classes = small_data.course.values.tolist()
c1, c2, c3 = small_classes
print(c2)
print(small_classes)

model = cp_model.CpModel()

# # dynamically set these?
# c1_time = model.NewIntervalVar()
# c2_time = model.NewIntervalVar()
# c3_time = model.NewIntervalVar()
#
# # constraint: credit limit
# model.AddLinearConstraint()
#
# # constraint: no class time overlap
# model.AddNoOverlap()
