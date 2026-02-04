INSIGHTS_TEMPLATE = """
## Key Insights from Related Tasks
The following are insights gathered during the execution of similar tasks. You may refer to them during your task execution to improve problem-solving accuracy.

{insights}
"""

POS_SHOTS_TEMPLATE = """
## Your Own Past Successes (Execution Patterns)
Here are examples of successful execution processes you've previously used on similar tasks.  
Pay special attention to the step-by-step procedures and strategies, especially when encountering obstacles:

{memory_few_shots}
"""

NEG_SHOTS_TEMPLATE = """
## Past Mistakes or Ineffective Strategies (What to Avoid)
Here are examples of less effective or failed execution patterns from your past attempts.  
Carefully review these examples to recognize common pitfalls, misunderstandings, or inefficient reasoning paths.  
When tackling new tasks, make sure to **avoid repeating these mistakes** and instead refine your approach based on what you've learned:

{memory_few_shots}
"""

EXTRACT_LATENT_PROMPT = """
# Current Task
{current_task}

# Examples of previous successful tasks related to this task
{text_memory}

You are acting as a {role}. Using no more than {k} tokens, extract the most relevant information from the memory above 
that will help you accomplish the current task effectively.
"""

EXTRACT_LATENT_PROMPT_FULL = """
# Current Task
{current_task}

# Examples of previous successful tasks related to this task
{text_memory}

# {role} response in each successful task above
{agent_message}

You are acting as a {role}. Using no more than {k} tokens, extract the most relevant information from the memory above 
that will help you accomplish the current task effectively.
"""