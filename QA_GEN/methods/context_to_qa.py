from typing import List
import re
from QA_GEN.base import QAPair
from QA_GEN.llms.ollama_chat import OllamaChat
from QA_GEN.method_register import Method


@Method(
    name="context_to_qa", 
    description="Expand context-only QA pairs to full QA pairs using LLM generation",
    applicable_stages=["data_expansion"],
    use_LLM=True,
    use_docker=False,
)
def context_to_qa(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Convert context-only QA pairs to full QA pairs by generating questions and answers.
    
    Args:
        qa_pairs (List[QAPair]): List of QA pairs to process
        config: Configuration object containing processing parameters
    
    Returns:
        List[QAPair]: List of expanded QA pairs
    """
    expanded_pairs = []
    context_pairs = [pair for pair in qa_pairs if pair.classify_id() == 1]  # Context-only type
    
    if not context_pairs:
        print("No context-only pairs found for expansion")
        return expanded_pairs
    
    print(f"Processing {len(context_pairs)} context-only pairs...")
    
    # Initialize LLM
    llm = OllamaChat()
    
    for i, context_pair in enumerate(context_pairs):
        try:
            # Generate full QA pairs using LLM
            prompt = f"""
You are an AI specialized in generating QA pairs from a given context.  
Generate a well-structured question and answer from the provided context.  
Output **only** in the following format:  

Q: <your question>  
A: <your answer>  

Context: {context_pair.context}  
"""
            
            response_text, response_info = llm(prompt=prompt)
            
            # Parse the response
            match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", response_text, re.DOTALL)
            
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                
                # Create new QA pair
                new_pair = QAPair(
                    question=question,
                    answer=answer,
                    context=context_pair.context
                )
                expanded_pairs.append(new_pair)
                
                if (i + 1) % 10 == 0:  # Progress logging
                    print(f"  Processed {i + 1}/{len(context_pairs)} context pairs")
                    
            else:
                print(f"  Warning: Failed to parse LLM response for pair {i+1}: {response_text[:100]}...")
                
        except Exception as e:
            print(f"  Error processing context pair {i+1}: {str(e)}")
            continue
    
    print(f"Successfully expanded {len(expanded_pairs)} QA pairs from {len(context_pairs)} context pairs")
    return expanded_pairs