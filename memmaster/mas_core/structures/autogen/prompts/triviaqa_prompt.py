ASSISTANT_SYSTEM_PROMPT_TEMPLATE = """
Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.
"""

ASSISTANT_USER_PROMPT_TEMPLATE = """
Below is the retrieved memory content and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
"""

USER_PROXY_SYSTEM_PROMPT_TEMPLATE = """
You will be given a question and an assistant's answer for that question. Follow the procedure below and produce outputs accordingly:

Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information.
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>.
You can search as many times as your want.
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.

Please consider the assistant agent's answer and provide your own answer.
"""

USER_PROXY_USER_PROMPT_TEMPLATE = """
Below is the relevant content retrieved from memory, the answer provided by the assistant, and the current task. Please provide your answer.

# Retrieved Content
{memory_content}

# Assistant Answer
{assistant_output}

# Current Task
{task_description}
"""

