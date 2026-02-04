# --- chatdev ---
CHATDEV_COMPRESS_SYSTEM_PROMPT="""
"""
CHATDEV_COMPRESS_USER_PROMPT="""
{trajectory}
"""

# --- generative ---
GENERATIVE_SYSTEM_PROMPT="""You are an agent designed to score the relevance between two pieces of text."""

GENERATIVE_USER_PROMPT = """
You are given:

1. A past case (which may be successful or failed):
{trajectory}

2. An ongoing task:
{task_description}

Your task is to evaluate **how relevant and useful the past case is for solving the ongoing task**.  
Use a **score from 1 to 10** (1 = not relevant at all, 10 = extremely relevant).

Only output the number. Do NOT include any explanations, text, or punctuation.

Score:
"""

# --- voyager ---
VOYAGER_SYSTEM_PROMPT = """
You are a helpful assistant that writes a description of the task resolution trajectory.
1) Summarize the trajectory in no more than 3 sentences.
2) Your response must be a single line of text with no line breaks.
"""

VOYAGER_USER_PROMPT = """
You are given the overall MAS trajectory on a task:
{trajectory}

Your job is to output a single-line summary describing the task and the trajectory (do not output anything irrelevant).
Your output:
"""

# --- gmemory ---
GMEMORY_FINETUNE_INSIGHTS_SUFFIX = dict(
    full = """Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES.""", 
    not_full = """"""
)

GMEMORY_FORMAT_RULES_TEMPLATE = """<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation.
"""

# 压缩 long trajectory
GMEMORY_COMPRESS_TRAJECTORY_SYSTEM_PROMPT = """
You are an agent skilled at extracting key steps.
Given a task and its successful execution trajectory, identify only the essential steps needed to complete the task while removing irrelevant or low-value actions.

Note:
- Strictly follow the original trajectory: do not add, reorder, or invent any steps.
- A successful trajectory may still contain incorrect actions; you must filter them out.
- Each retained step must be at the smallest meaningful unit (atomic actions; do not merge or further split).
- Keep your output concise.
"""
GMEMORY_COMPRESS_TRAJECTORY_USER_PROMPT = """
## Task Trajectory
{task_trajectory}

Your output: 
"""

GMEMORY_COMPARE_GENERATION_SYSTEM_PROMPT = """
You are an advanced reasoning agent that derives general rules from examples.
You will receive one successful trial and one failed trial.

Your goal:
- Compare the positive and negative examples to extract insights that help avoid similar mistakes.
- The insights must be concise and expressed as high-level reasoning principles, not tied to specific items or tasks.
"""

GMEMORY_COMPARE_GENERATION_USER_PROMPT = """
## Successful Trial
{pos_shot}

## Failed Trial
{neg_shot}

## Existing Rules
{existing_rules}

## Your task:
Compare the successful and failed trials. Update the rule list by adding, editing, removing, or agreeing so that the rules become general, high-level reasoning guidelines for avoiding similar failures. 
Output only in the required format: 
""" + GMEMORY_FORMAT_RULES_TEMPLATE

GMEMORY_SUMMARIZE_GENERATION_SYSTEM_PROMPT = """
You are an advanced reasoning agent capable of adding, editing, or removing rules from an existing rule set by forming new critiques of past task trajectories.
You will receive a set of successful trajectories.

Your goal:
- Summarize general insights from these successful trajectories to guide future problem solving in similar contexts.
- Ensure the insights are concise, expressed as high-level reasoning principles, and not tied to specific items or tasks.
"""

GMEMORY_SUMMARIZE_GENERATION_USER_PROMPT = """
## Successful Trials
{pos_shots}

## Existing Rules
{existing_rules}

Your task:
Based on the successful trials and existing rules, update the rule set (add, edit, remove, or agree).  
Ensure the final rules are high-level, general insights that guide better Thought and Action across diverse tasks.  
Follow the requird output format: 
""" + GMEMORY_FORMAT_RULES_TEMPLATE
