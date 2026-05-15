from google import genai

# ✅ Configure client
client = genai.Client(api_key="YOUR_API_KEY")

memory = {}

def store(key, value):
    memory[key] = value
    return f"Stored {key} = {value}"

# ✅ STEP 1 — PLAN STEPS
def plan_steps(goal):
   
    prompt = f"""
Break this goal into steps.

IMPORTANT:
- If calculation is needed, create actual numbers
- Store values before calculation

Goal: {goal}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# ✅ STEP 2 — TOOLS
def calculator(expr):
    try:
        
        for key in memory:
            expr = expr.replace(key, str(memory[key]))

        result = eval(expr)
        memory["last_result"] = result

        return result

    except:
        return "Invalid math input"

def planner(task):
    return f"Planning done for: {task}"


# ✅ STEP 3 — DECIDE ACTION
def decide_action(step):
    prompt = f"""
You are an AI agent.

Decide best tool:
- calculator (for math)
- planner (for tasks)
- store (to store values)


Respond ONLY in this exact format:
action: <tool>
input: <input>

Step: {step}
"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


# ✅ STEP 4 — SAFE PARSER (ROBUST)
def parse_response(text):
    action = None
    inp = None

    lines = text.lower().split("\n")

    for line in lines:
        if "action" in line:
            action = line.split(":")[-1].strip()
        elif "input" in line:
            inp = line.split(":")[-1].strip()

    return action, inp


# ✅ STEP 5 — EXECUTION
def execute(action, inp):
    if action == "calculator":
        return calculator(inp)

    elif action == "planner":
        return planner(inp)

    elif action == "store":
        parts = inp.split(",")
        key = parts[0].strip()
        value = parts[1].strip()

        return store(key, value)

    else:
        return "Unknown action"



# ✅ STEP 6 — MAIN AGENT LOOP
def agent(goal):
    print("\n🎯 Goal:", goal)

    steps = plan_steps(goal)
    print("\n🧠 Planned Steps:\n", steps)

    for step in steps.split("\n"):
        step = step.strip()

        if step == "":
            continue

        # ✅ remove numbering like "1."
        if "." in step:
            step = step.split(".", 1)[1].strip()

        print("\n➡️ Processing step:", step)

        decision = decide_action(step)
        print("🤖 Decision:\n", decision)

        action, inp = parse_response(decision)

        # ✅ handle parsing failure
        if not action or not inp:
            print("⚠️ Parsing failed, skipping step")
            continue

        result = execute(action, inp)
        print("✅ Result:", result)

print("📦 Memory:", memory)
# ✅ ENTRY POINT
if __name__ == "__main__":
    goal = input("Enter your goal: ")
    agent(goal)
