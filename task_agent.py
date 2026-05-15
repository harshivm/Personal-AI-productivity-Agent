from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tasks = [
    "study for math exam",
    "complete assignment",
    "go to gym",
    "prepare presentation",
    "watch netflix",
    "revise notes",
    "write project report",
    "play games"
]
labels = [
    "HIGH",
    "MEDIUM",
    "LOW",
    "HIGH",
    "LOW",
    "HIGH",
    "MEDIUM",
    "LOW"
]
time = [
    "3",
    "2",
    "1",
    "3",
    "1",
    "3",
    "2",
    "1"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tasks)
model = LogisticRegression()
model.fit(X,labels)
model2 = LogisticRegression()
model2.fit(X,time)

def get_user_goal():
    goal = input("Please enter your task: ")
    return goal

def simple_task_breakdown(goal):
    tasks = goal.split("and")
    tasks = [task.strip() for task in tasks]
    return tasks


def assign_priority(task):
    task_prio_vec = vectorizer.transform([task])
    prediction = model.predict(task_prio_vec)
    return prediction[0]
    
def assign_time(task):
    task_time_vec = vectorizer.transform([task])
    prediction = model2.predict(task_time_vec)   
    return prediction[0]



goal = get_user_goal()
tasks = simple_task_breakdown(goal)


print("Tasks identified")
for t in tasks:
    print("-",t)

print("Task Priorities:")
for t in tasks:
    priority = assign_priority(t)
    print(f"-{t},[{priority}]")

print("Estimated time")
for t in tasks:
    priority = assign_priority(t)
    time = assign_time(t)
    print(f"{t}| {priority} | {time}")



