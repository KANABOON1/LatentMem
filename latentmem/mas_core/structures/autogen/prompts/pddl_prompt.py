ASSISTANT_SYSTEM_PROMPT_TEMPLATE = """
You are a problem-solving agent.
"""

ASSISTANT_USER_PROMPT_TEMPLATE = """
Below is the relevant content retrieved from memory and the current task. Please provide your response.

# Retrieved Content
{memory_content}

# Current Task
{task_description}
"""

USER_PROXY_SYSTEM_PROMPT_TEMPLATE = """
You are a problem-solving agent.
"""

USER_PROXY_USER_PROMPT_TEMPLATE = """
Below is the relevant content retrieved from memory, the strategy provided by the assistant, and the current task requirements. Please provide your response.

# Retrieved Content
{memory_content}

# Implementation Strategy
{assistant_output}

# Current Task
{task_description}
"""

