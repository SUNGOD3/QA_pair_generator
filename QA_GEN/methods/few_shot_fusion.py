from typing import List
import re
from QA_GEN.base import QAPair
from QA_GEN.llms.ollama_chat import OllamaChat
from QA_GEN.method_register import Method

@Method(
    name="few_shot_fusion",
    description="Generate new QA pairs by providing a few similar examples to the LLM",
    applicable_stages=["data_fusion"],
    use_LLM=True
)
def few_shot_fusion(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Create new QA pairs by showing a few examples to the LLM and asking it to generate
    new content following the same pattern or structure.
    
    Args:
        qa_pairs (List[QAPair]): List of QA pairs to use as examples
        
    Returns:
        List[QAPair]: Newly generated QA pairs
    """
    if len(qa_pairs) < 2:
        print("Need at least 2 QA pairs for few-shot fusion")
        return []
    # Get the source pair and example pairs
    source_pair = qa_pairs[0]
    example_pairs = qa_pairs[1:]
    
    # Prepare examples for few-shot learning
    examples_text = ""
    for i, example in enumerate(example_pairs, 1):
        examples_text += f"Example {i}:\n"
        examples_text += f"Context: {example.context}\n"
        examples_text += f"Question: {example.question}\n"
        examples_text += f"Answer: {example.answer}\n\n"
    
    # Create prompt for LLM
    llm = OllamaChat()
    prompt = f"""
You are an expert at creating educational question-answer pairs.
I'll show you some example QA pairs, and I want you to create a new, original QA pair that follows the same pattern, style, and difficulty level.

Here are some examples:

{examples_text}

Now, based on these examples and using the following context as inspiration, create a new, original QA pair:

Context: {source_pair.context}

Please provide your response in this format:
Context: (a variation of the provided context, modified to be original)
Question: (your original question)
Answer: (your original answer)
"""

    # Generate new QA pair
    response_text, response_info = llm(prompt=prompt)
    
    # Parse the response
    context_match = re.search(r"Context:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    question_match = re.search(r"Question:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    answer_match = re.search(r"Answer:\s*(.*?)(?:\n|$)", response_text, re.DOTALL)
    
    if context_match and question_match and answer_match:
        new_context = context_match.group(1).strip()
        new_question = question_match.group(1).strip()
        new_answer = answer_match.group(1).strip()
        
        # Create a new QA pair
        new_pair = QAPair(
            question=new_question,
            answer=new_answer,
            context=new_context
        )
        
        return [new_pair]
    else:
        print("Failed to parse the LLM response for few-shot fusion")
        return []