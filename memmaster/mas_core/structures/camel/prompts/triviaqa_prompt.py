# --- user proxy prompt ---
USERPROXY_SYSTEM_PROMPT_TEMPLATE = """
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
"""

USERPROXY_USER_PROMPT_TEMPLATE = """
Below is the retrieved memory content and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
"""

# --- actor prompt ---
ACTOR_SYSTEM_PROMPT_TEMPLATE = """
You will be given a question and an user proxy agent's answer for that question. Follow the procedure below and produce outputs accordingly:

Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

Please consider the user proxy agent's answer and provide your own answer.
"""

ACTOR_USER_PROMPT_TEMPLATE = """
Below is the relevant content retrieved from memory, the answer provided by the user proxy agent, and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# User Proxy Answer
{userproxy_output}

# Current Task
{task_description}
"""

# --- critic prompt ---
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

# --- summarizer prompt ---
SUMMARIZER_SYSTEM_PROMPT_TEMPLATE = """
You will be given a question, the responses produced by actor agents, and the corresponding feedback from critic agents for that question. Follow the procedure below and produce outputs accordingly:

Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
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