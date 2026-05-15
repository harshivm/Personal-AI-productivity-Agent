from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

tasks = [
    "interview preparation",
    "study for math exam",
    "complete assignment",
    "go to gym",
    "prepare presentation",
    "watch netflix",
    "revise notes",
    "write project report",
    "play games",
    "sleep",
    "attend interview",
    "climb mountain",
    "do workout",
    "buy groceries"

]
labels = [
    "HIGH",
    "HIGH",
    "MEDIUM",
    "LOW",
    "HIGH",
    "LOW",
    "HIGH",
    "MEDIUM",
    "LOW",
    "LOW",
    "HIGH",
    "MEDIUM",
    "MEDIUM",
    "LOW"

]
time = [
    "3",
    "3",
    "2",
    "1",
    "3",
    "1",
    "3",
    "2",
    "1",
    "1",
    "3",
    "2",
    "2",
    "1"
    
]

vectorizer = TfidfVectorizer() #variable for vectorization
X = vectorizer.fit_transform(tasks) # x is already vectorised

# here i have used 2 model, one for priority and other for time
model = LogisticRegression()
model.fit(X,labels)

model2 = LogisticRegression()
model2.fit(X,time)

# slitting the data into training and testing 
X_train, X_test, y_train, y_test, time_train, time_test= train_test_split(tasks, labels, time, test_size=0.3, random_state=1)

# vectorization of training and testing data
# vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# training the model on training data
model.fit(X_train_vec, y_train)
model2.fit(X_train_vec, time_train)

# testing model on testing data
y_pred = model.predict(X_test_vec)

# accuracy is calculated
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# prediction in detail
print("detailed predictions:")
for task, true_label, pred in zip(X_test, y_test, y_pred):
    print(f"Task: {task}, True Label: {true_label}, Predicted Label: {pred}")

# for input by user
def get_user_goal():
    goal = input("Please enter your task: ")
    return goal

# for task
def simple_task_breakdown(goal):
    tasks = goal.split("and")
    tasks = [task.strip() for task in tasks]
    return tasks

# for priority
def assign_priority(task):
    task_prio_vec = vectorizer.transform([task])
    prediction = model.predict(task_prio_vec)
    return prediction[0]

# for time   
def assign_time(task):
    task_time_vec = vectorizer.transform([task])
    # prediction = model2.predict(task_time_vec)   
    return model2.predict(task_time_vec)[0]


# main function
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



