from typing import List
import re
from QA_GEN.base import QAPair
from QA_GEN.method_register import Method

@Method(
    name="refine_repeated_content", 
    description="Refines QA pairs by removing consecutively repeated words or phrases",
    applicable_stages=["data_filter"],
    use_LLM=False
)
def refine_repeated_content(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Refines QA pairs by removing consecutively repeated words or phrases.
    
    Args:
        qa_pairs: A list of QAPair objects to refine
        config: Configuration dictionary containing parameters
            - 'min_repetitions': Minimum number of consecutive repetitions to trigger cleanup (default: 2)
    
    Returns:
        List[QAPair]: The same QAPair objects with refined content
    """
    # Get config parameters
    min_repetitions = config.get('min_repetitions', 2)
    
    for qa_pair in qa_pairs:
        # Refine context if it exists
        if qa_pair.context:
            qa_pair.set_context(clean_repeated_content(qa_pair.context, min_repetitions))
        
        # Refine question if it exists
        if qa_pair.question:
            qa_pair.set_question(clean_repeated_content(qa_pair.question, min_repetitions))
            
        # Refine answer if it exists
        if qa_pair.answer:
            qa_pair.set_answer(clean_repeated_content(qa_pair.answer, min_repetitions))
    
    return qa_pairs

def clean_repeated_content(text: str, min_repetitions: int = 2) -> str:
    """
    Remove consecutively repeated words or phrases from text.
    
    Args:
        text: The text to clean
        min_repetitions: Minimum number of consecutive repetitions to trigger cleanup
    
    Returns:
        str: Cleaned text with repetitions removed
    """
    if not text:
        return text
        
    # Function to find and remove consecutive repetitions of the same word
    def clean_repeated_words(text):
        # Match the same word repeated multiple times (case insensitive)
        pattern = r'\b(\w+)(?:\s+\1){' + str(min_repetitions-1) + r',}\b'
        
        # Find all matches
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        
        # Process text from end to start to avoid index issues when replacing
        for match in reversed(matches):
            word = match.group(1)
            start, end = match.span()
            # Replace with a single instance of the word
            text = text[:start] + word + text[end:]
            
        return text
    
    # Function to find and remove longer repeating phrases
    def clean_repeated_phrases(text, phrase_length):
        if phrase_length < 2:
            return text
            
        words = text.split()
        if len(words) < phrase_length * min_repetitions:
            return text
            
        i = 0
        result = []
        while i < len(words):
            # Get current phrase
            if i + phrase_length > len(words):
                result.extend(words[i:])
                break
                
            current_phrase = words[i:i+phrase_length]
            
            # Count how many times this phrase repeats
            repetitions = 1
            while i + (repetitions * phrase_length) + phrase_length <= len(words):
                next_chunk = words[i+(repetitions*phrase_length):i+(repetitions*phrase_length)+phrase_length]
                if current_phrase == next_chunk:
                    repetitions += 1
                else:
                    break
            
            # If phrase repeats more than min_repetitions, add it only once
            if repetitions >= min_repetitions:
                result.extend(current_phrase)
                i += repetitions * phrase_length
            else:
                result.append(words[i])
                i += 1
                
        return ' '.join(result)
    
    # Apply cleaning for individual repeated words
    cleaned = clean_repeated_words(text)
    
    # Apply cleaning for repeated phrases (up to 5 words long)
    for phrase_length in range(2, 6):
        cleaned = clean_repeated_phrases(cleaned, phrase_length)
        
    return cleaned