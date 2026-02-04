USERPROXY_SYSTEM_PROMPT_TEMPLATE = """
You are a problem-solving agent.
"""

USERPROXY_USER_PROMPT_TEMPLATE = """
Below is the relevant content retrieved from memory and the current task. Please provide your response in accordance with the task requirements.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
"""

ACTOR_SYSTEM_PROMPT_TEMPLATE = """
You will be given a question and an user proxy agent's answer for that question. Please consider the user proxy agent's answer and provide your own answer.
"""

ACTOR_USER_PROMPT_TEMPLATE = """
Below is the relevant content retrieved from memory, the answer provided by the user proxy agent, and the current task. Please provide your response in accordance with the task requirements.

# Retrieved Content
{memory_content}

# User Proxy Answer
{userproxy_output}

# Current Task
{task_description}
"""

CRITIC_SYSTEM_PROMPT_TEMPLATE = """
You are a strategy evaluator. Your task is to review the response provided by the actor agent for the current problem.

- If you believe the actor agent's response is correct and has no issues, reply only with: "Agree".
- If you believe the actor agent's response has issues, provide brief and concise feedback only (keep your response short and within 3 sentences).
"""

CRITIC_USER_PROMPT_TEMPLATE = """
Below are the relevant contents retrieved from memory, the response given by the actor agent, and the requirements of the current task. Please provide your review.

# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Current Task
{task_description}
"""

SUMMARIZER_SYSTEM_PROMPT_TEMPLATE = """
You will be given a question, the responses produced by actor agents, and the corresponding feedback from critic agents for that question. Please provide your response in accordance with the task requirements.
"""

SUMMARIZER_USER_PROMPT_TEMPLATE = """
# Retrieved Content
{memory_content}

# Actor Output
{actor_output}

# Critic Feedback
{critic_output}

# Current Task
{task_description}
"""