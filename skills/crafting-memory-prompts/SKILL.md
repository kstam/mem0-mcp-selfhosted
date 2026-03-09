---
name: crafting-memory-prompts
description: Use when setting up mem0 memory for a new use case, when memory is extracting irrelevant facts or missing important ones, or when the user wants to customize what their memory system remembers
---

# Crafting Memory Prompts

## Overview

Guide a user through creating a custom `custom_fact_extraction_prompt` for mem0. The prompt controls what gets remembered vs discarded. mem0's default prompt is tuned for personal assistants — any other use case (executive assistant, coding, customer support) needs a tailored prompt.

## When to Use

- Setting up mem0 for a new project or use case
- Memory is noisy (storing irrelevant facts) or too quiet (missing important ones)
- User says "customize what mem0 remembers"
- Switching domains (e.g., personal → work assistant)

## How mem0 Uses This Prompt

The `custom_fact_extraction_prompt` is passed as the **system message** to the LLM. The **user message** is `"Input:\n{conversation_text}"`. The LLM must return `{"facts": [...]}` — a JSON object with a `facts` key containing a list of strings. Empty list = nothing to remember.

When set, it replaces BOTH the user-memory and agent-memory defaults.

## Interactive Process

Follow these steps **in order**, one question per message. Do not skip steps.

### Step 1: Define the Role

Ask: **"Describe the role of this memory system in one sentence."**

Examples the user might say:
- "Executive assistant that remembers my work context"
- "Coding assistant that tracks project decisions and preferences"
- "Customer support agent that remembers account details"

This becomes the opening line of the prompt.

### Step 2: Define Memory Categories

Ask: **"What types of information should this system remember? List the categories that matter."**

Suggest starter categories based on their role to help them think. For example, if they said "executive assistant":
- Relationships and org structure (who reports to whom, key stakeholders)
- Preferences and habits (communication style, meeting preferences)
- Plans, commitments, and deadlines
- Ideas, visions, and strategic thinking
- Tool and service preferences
- Structured identifiers (Confluence page IDs, Jira project keys, Slack channel IDs, account IDs, API endpoints)

**Always suggest "Structured Identifiers" as a category.** These are high-value, easy to lose, and most users forget to include them.

Let the user refine — add, remove, or reword categories. Each becomes a numbered item in the prompt.

### Step 3: Define Exclusion Rules

Ask: **"What should be explicitly ignored? What would be noise in your memory?"**

Suggest common exclusions based on their domain:
- Greetings and small talk ("Hi", "Thanks", "How are you")
- Ephemeral status updates ("I'm in a meeting right now")
- Verbatim code blocks or file contents
- Routine timestamps ("I had lunch at noon")
- Information that's already captured in external systems (if they use Jira for task tracking, don't duplicate task status in memory)

These become the "Do NOT extract" section.

### Step 4: Create Few-Shot Examples

Ask: **"Give me 2-3 example conversations that are typical for your use case. I'll turn them into few-shot examples."**

From each conversation, generate:
- **Positive case**: Input that should produce facts → `{"facts": ["fact1", "fact2"]}`
- **Negative case**: Input that should produce nothing → `{"facts": []}`

**Requirements:**
- Minimum 3 positive examples (showing fact extraction)
- Minimum 2 negative examples (showing empty results)
- Include at least one example with structured identifiers
- Show the user the generated examples and ask if they look right

### Step 5: Generate and Save

Assemble the prompt using this structure:

```
You are a {ROLE}, specialized in accurately storing {DOMAIN_SUMMARY}.
Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts.
This allows for easy retrieval and personalization in future interactions. Below are the types of information you need to focus on and the detailed instructions on how to handle the input data.

# [IMPORTANT]: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGES. DO NOT INCLUDE INFORMATION FROM ASSISTANT OR SYSTEM MESSAGES.

Types of Information to Remember:

1. {Category}: {Description}
2. {Category}: {Description}
...

Do NOT extract:
- {Exclusion 1}
- {Exclusion 2}
...

Here are some few-shot examples:

{All positive and negative examples in User/Assistant/Output format}

Return the facts and preferences in a JSON format as shown above.

Remember the following:
- Generate facts solely from user messages. Do not pick anything from assistant or system messages.
- Today's date is {current_date_placeholder}.
- If you do not find anything relevant, return an empty list: {"facts": []}
- Make sure to return the response in JSON format with a key "facts" and a list of strings as the value.
- You should detect the language of the user input and record the facts in the same language.
- Each fact should be a concise, standalone statement. Prefer specific details over vague summaries.
- When a fact contains an identifier (ID, URL, key), always include it verbatim in the fact string.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user, if any, from the conversation and return them in the json format as shown above.
```

**Date placeholder**: Use the literal string `{current_date}` — mem0 does NOT template this. Instead, the skill should note that the prompt is static. If the user needs the date, they should include it as a general instruction.

**Save to**: `prompts/fact_extraction.txt` in the project root (create directory if needed).

**Tell the user**: Set `MEM0_CUSTOM_FACT_PROMPT_FILE=prompts/fact_extraction.txt` in your `.env` file. Restart the MCP server to pick up the new prompt.

## Quality Checklist

Before saving, verify the generated prompt:
- [ ] Has a clear role statement in the opening line
- [ ] Lists all agreed-upon categories with descriptions
- [ ] Has explicit exclusion rules
- [ ] Has 3+ positive few-shot examples
- [ ] Has 2+ negative few-shot examples (returning empty facts)
- [ ] At least one example includes a structured identifier
- [ ] Includes the JSON format instruction (`{"facts": [...]}`)
- [ ] Includes the "extract from user messages only" instruction
- [ ] Each few-shot example uses the User/Assistant/Output format

## Common Mistakes

| Mistake | Fix |
|---------|-----|
| Categories too broad ("Remember everything important") | Be specific: "Org structure: who reports to whom, team compositions, role changes" |
| No negative examples | Always include 2+ examples that return `{"facts": []}` |
| Missing structured identifiers category | Always suggest it — IDs, URLs, keys are high-value and easy to lose |
| Facts are too verbose | Instruct: "Each fact should be a concise, standalone statement" |
| Forgetting exclusion rules | Without them, memory fills with greetings and small talk |
