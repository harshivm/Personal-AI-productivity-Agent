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

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tasks)
model = LogisticRegression()
model.fit(X,labels)

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
    
    
def estimated_time(task):
    task = task.lower()
    if "exam" in task:
        return "3 hours"
    elif "assignment" in task:
        return "2 hours"
    else:
        return "1 hour"


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
    time = estimated_time(t)
    print(f"{t}| {priority} | {time}")



