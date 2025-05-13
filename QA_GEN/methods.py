#methods.py
from typing import Any, Callable, Dict, List
from .base import QAPair
from .llms.oai_chat import OpenAIChat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import random

class Method:
    """
    Decorator class to automatically register methods with metadata.
    Stores all discovered methods with their associated information.
    """
    # Class-level dictionary to store all discovered methods
    _methods: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self, 
        name: str, 
        description: str, 
        applicable_stages: List[str], # List[str] = ["init_setup", "data_expansion", "build_knowledge_graph", "data_fusion", "data_filter", "data_augmentation"]
        use_LLM: bool = False,
        complexity: str = None  # Optional field for describing complexity
    ):
        """
        Initialize a method with its metadata.
        
        Args:
            name (str): Unique name of the method
            description (str): Detailed description of the method
            applicable_stages (List[str]): Stages where this method can be applied
            use_LLM (bool, optional): Whether the method uses LLM. Defaults to False.
            complexity (str, optional): Description of method's time/space complexity. Optional field.
        """
        self.name = name
        self.description = description
        self.applicable_stages = applicable_stages
        self.use_LLM = use_LLM
        self.complexity = complexity

    def __call__(self, func: Callable):
        """
        Register the method when the decorator is applied.
        
        Args:
            func (Callable): The method function to be registered
        
        Returns:
            Callable: The original function
        """
        # Store method information
        method_info = {
            'func': func,
            'name': self.name,
            'description': self.description,
            'applicable_stages': self.applicable_stages,
            'use_LLM': self.use_LLM
        }
        
        # Only add complexity if provided
        if self.complexity is not None:
            method_info['complexity'] = self.complexity
            
        Method._methods[self.name] = method_info
        return func

    @classmethod
    def get_methods(cls) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve all registered methods.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of registered methods
        """
        return cls._methods

    @classmethod
    def get_methods_for_stage(cls, stage: str) -> Dict[str, Dict[str, Any]]:
        """
        Get methods applicable to a specific stage.
        
        Args:
            stage (str): Pipeline stage to filter methods for
        
        Returns:
            Dict[str, Dict[str, Any]]: Methods applicable to the stage
        """
        return {
            name: method for name, method in cls._methods.items()
            if stage in method['applicable_stages']
        }



@Method(
    name="context_to_qa", 
    description="Expand context-only QA pairs to full QA pairs",
    applicable_stages=["data_expansion"],
    use_LLM=True,
    complexity="O(n) where n is the number of context pairs; LLM calls may impact performance"
)
def context_to_qa(qa_pairs: List[QAPair], config) -> List[QAPair]:
    expanded_pairs = []
    for context_pair in qa_pairs:
        if context_pair.classify_id() == 1:  # Context-only type
            llm = OpenAIChat()
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
            match = re.search(r"Q:\s*(.*?)\s*A:\s*(.*)", response_text, re.DOTALL)
            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()
                new_pair = QAPair(
                    question=question,
                    answer=answer,
                    context=context_pair.context
                )
                expanded_pairs.append(new_pair)
            else:
                print("Parsing failed:", response_text)
    return expanded_pairs


@Method(
    name="key_sentences_question",
    description="Generate only questions based on key sentences",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def key_sentences_question(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        sentences =  qa_pair.context_keywords
        if not sentences:
            return []
        key_sentence = random.choice(sentences)
        question = f"Based on the content, what does '{key_sentence}' mean?"
        qa_pair.set_question(question)
        rt_pairs.append(qa_pair)
    return rt_pairs

@Method(
    name="generate_summary_question",
    description="Generate summary questions, where the question is about the summary of the content",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def generate_summary_question(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        question = f"Based on the content, what is the key information summarized?"
        qa_pair.set_question(question)
        rt_pairs.append(qa_pair)
    return rt_pairs

@Method(
    name="generate_summary_qa",
    description="Generate summary-type QA pairs, where the question is about the summary of the content",
    applicable_stages=["data_expansion"],
    use_LLM=True
)
def generate_summary_qa(qa_pairs, config):
    rt_pairs = []
    llm = OpenAIChat()
    question = f"Based on the content, what is the key information summarized?"
    for qa_pair in qa_pairs:
        qa_pair.set_question(question)
        response_text, response_info = llm(prompt=qa_pair.context + '\n' + question)
        summarized_text = response_text.strip()
        qa_pair.set_answer(summarized_text)
        rt_pairs.append(qa_pair)
    return rt_pairs

@Method(
    name="generate_fill_in_blank",
    description="Generate fill-in-the-blank type QA pairs",
    applicable_stages=["data_expansion"],
    use_LLM=False
)
def generate_fill_in_blank(qa_pairs, config):
    rt_pairs = []
    for qa_pair in qa_pairs:
        context = qa_pair.context
        words = word_tokenize(context)
        words = [w for w in words if w.isalnum() and w.lower() not in stopwords.words("english")]
        if not words:
            return []
        blank_word = random.choice(words)
        question = context.replace(blank_word, "____")
        qa_pair.set_question(question)
        qa_pair.set_answer(blank_word)
        rt_pairs.append(qa_pair)
    return rt_pairs

print("Methods registered:")
for name, method in Method.get_methods().items():
    print(f" - {name}: {method['description']}")

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
    llm = OpenAIChat()
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
    
@Method(
    name="cross_domain_fusion",
    description="Fuse two moderately related QA pairs to create a new hybrid QA pair that bridges both domains",
    applicable_stages=["data_fusion"],
    use_LLM=True
)
def cross_domain_fusion(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Create a new QA pair by fusing knowledge from two moderately related QA pairs.
    This method works best when the input pairs have some conceptual overlap but
    are from different domains or contexts.
    
    Args:
        qa_pairs (List[QAPair]): List of QA pairs to fuse (ideally 2 pairs)
        
    Returns:
        List[QAPair]: Newly generated hybrid QA pair
    """
    # Need exactly 2 QA pairs for this fusion method
    if len(qa_pairs) != 2:
        print("Cross-domain fusion requires exactly 2 QA pairs")
        return []
    
    pair1, pair2 = qa_pairs
    
    # Create prompt for LLM
    llm = OpenAIChat()
    prompt = f"""
You are an expert at creating content that bridges multiple domains or topics.
I'll show you two different QA pairs that have some moderate relationship, and I want you to create a new, 
original QA pair that creatively combines elements from both sources.

First QA Pair:
Context: {pair1.context}
Question: {pair1.question}
Answer: {pair1.answer}

Second QA Pair:
Context: {pair2.context}
Question: {pair2.question}
Answer: {pair2.answer}

Create a new QA pair that bridges these two topics or domains. The new QA pair should:
1. Identify a conceptual link or relationship between these two areas
2. Create something novel that couldn't be derived from either source alone
3. Be factually accurate

Please provide your response in this format:
Context: (a bridge context that connects elements from both sources)
Question: (your original question that spans both domains)
Answer: (your original answer that incorporates knowledge from both sources)
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
        print("Failed to parse the LLM response for cross-domain fusion")
        return []
    

@Method(
    name="filter_similar_pairs", 
    description="Filter out similar QA pairs by randomly sampling and comparing pairs using LLM",
    applicable_stages=["data_filter"],
    use_LLM=True,
)
def filter_similar_pairs(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Filters out similar QA pairs by randomly sampling K pairs and comparing them using LLM.
    For each comparison, if pairs are found to be similar, the shorter one is removed.
    
    Args:
        qa_pairs: A list of QAPair objects to filter
        config: Configuration dictionary containing parameters like 'k' for sampling times
    
    Returns:
        List[QAPair]: Filtered list of QAPair objects with similar pairs removed
    """
    if len(qa_pairs) <= 1:
        return qa_pairs
    
    # Get sampling parameter K from config
    k = config.get('k', min(10, len(qa_pairs) * (len(qa_pairs) - 1) // 2))
    
    # Create a copy to avoid modifying the original list during iteration
    remaining_pairs = qa_pairs.copy()
    pairs_to_remove = set()
    
    # Keep track of already compared pairs to avoid duplicates
    compared_pairs = set()
    
    # Initialize LLM
    llm = OpenAIChat()
    
    # Sample K times
    sample_count = 0
    max_attempts = k * 2  # To avoid infinite loops if we run out of unique pairs
    attempt_count = 0
    
    while sample_count < k and attempt_count < max_attempts and len(remaining_pairs) > 1:
        attempt_count += 1
        
        # Get indices of two random pairs that haven't been removed yet
        valid_indices = [i for i, pair in enumerate(remaining_pairs) if i not in pairs_to_remove]
        if len(valid_indices) < 2:
            break
        
        idx1, idx2 = random.sample(valid_indices, 2)
        pair1, pair2 = remaining_pairs[idx1], remaining_pairs[idx2]
        
        # Create a unique identifier for this comparison
        pair_id = tuple(sorted([pair1.id, pair2.id]))
        
        # Skip if we've already compared these pairs
        if pair_id in compared_pairs:
            continue
        
        # Mark this pair as compared
        compared_pairs.add(pair_id)
        sample_count += 1
        
        # Prepare prompt to check similarity
        prompt = f"""
Compare these two QA pairs and determine if they are semantically similar or duplicates.
Answer with only 'YES' if they are similar enough that one should be removed, or 'NO' if they are sufficiently different.

QA Pair 1:
Context: {pair1.context or "N/A"}
Question: {pair1.question or "N/A"}
Answer: {pair1.answer or "N/A"}

QA Pair 2:
Context: {pair2.context or "N/A"}
Question: {pair2.question or "N/A"}
Answer: {pair2.answer or "N/A"}

Are these QA pairs similar or duplicates? (YES/NO)
"""
        
        # Ask LLM if pairs are similar
        response, _ = llm(prompt=prompt)
        
        # Check if response indicates similarity
        if "YES" in response.upper():
            # Calculate length of each pair (total characters in context+question+answer)
            len1 = len((pair1.context or "") + (pair1.question or "") + (pair1.answer or ""))
            len2 = len((pair2.context or "") + (pair2.question or "") + (pair2.answer or ""))
            
            # Remove the shorter pair
            if len1 <= len2:
                pairs_to_remove.add(idx1)
            else:
                pairs_to_remove.add(idx2)
    
    # Create final filtered list
    filtered_pairs = [pair for i, pair in enumerate(remaining_pairs) if i not in pairs_to_remove]
    
    return filtered_pairs



@Method(
    name="filter_exact_substring", 
    description="Filter out QA pairs where one contains a substantial exact substring of another",
    applicable_stages=["data_filter"],
    use_LLM=False
)
def filter_exact_substring(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Filters out QA pairs where one contains a substantial exact substring of another.
    For each match, the longer QA pair is removed.
    
    Args:
        qa_pairs: A list of QAPair objects to filter
        config: Configuration dictionary containing parameter 'k' for minimum substring length
    
    Returns:
        List[QAPair]: Filtered list of QAPair objects with redundant pairs removed
    """
    if len(qa_pairs) <= 1:
        return qa_pairs
    
    # Get minimum substring length from config, default to 50 characters
    min_substring_length = config.get('k', 50)
    
    # Create a copy to avoid modifying the original list during iteration
    pairs_to_remove = set()
    
    # Helper function to check for substring match
    def check_substring_match(text1: str, text2: str) -> bool:
        """Check if one text contains a substantial substring of the other."""
        if not text1 or not text2:
            return False
            
        # Check if text1 contains text2 as a substantial substring
        if text1 != text2 and len(text2) >= min_substring_length and text2 in text1:
            return True
            
        # Check if text2 contains text1 as a substantial substring
        if text1 != text2 and len(text1) >= min_substring_length and text1 in text2:
            return True
            
        return False
    
    # Compare all pairs
    for i in range(len(qa_pairs)):
        if i in pairs_to_remove:
            continue
            
        for j in range(i+1, len(qa_pairs)):
            if j in pairs_to_remove:
                continue
                
            pair1, pair2 = qa_pairs[i], qa_pairs[j]
            
            # Extract all text fields from each pair
            texts1 = [pair1.context or "", pair1.question or "", pair1.answer or ""]
            texts2 = [pair2.context or "", pair2.question or "", pair2.answer or ""]
            
            # Check each combination of text fields for substring matches
            found_match = False
            for t1 in texts1:
                for t2 in texts2:
                    if check_substring_match(t1, t2):
                        found_match = True
                        break
                if found_match:
                    break
            
            # If a match is found, remove the longer pair
            if found_match:
                # Calculate total length of each pair
                len1 = sum(len(t) for t in texts1)
                len2 = sum(len(t) for t in texts2)
                
                # Remove the longer pair (prioritize keeping shorter content)
                if len1 > len2:
                    pairs_to_remove.add(i)
                else:
                    pairs_to_remove.add(j)
    
    # Create final filtered list
    filtered_pairs = [pair for idx, pair in enumerate(qa_pairs) if idx not in pairs_to_remove]
    
    return filtered_pairs

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


@Method(
    name="word_swap_augmentation", 
    description="Augment QA pairs by randomly swapping pairs of words in questions n times",
    applicable_stages=["data_augmentation"],
    use_LLM=False,
    complexity="O(m*n) where m is the number of QA pairs and n is the number of swaps per pair"
)
def word_swap_augmentation(qa_pairs: List[QAPair], config) -> List[QAPair]:
    """
    Augment QA pairs by randomly swapping words in questions.
    
    Args:
        qa_pairs: List of QAPair objects to augment
        config: Configuration dictionary containing parameters:
            - num_swaps: Number of word pairs to swap per question (default: 1)
            - augmentation_factor: Number of augmented versions to create per QA pair (default: 3)
    
    Returns:
        List of new QAPair objects with swapped word questions
    """
    # Set default values for configuration
    num_swaps = config.get("num_swaps", 1)
    augmentation_factor = config.get("augmentation_factor", 1)
    
    augmented_pairs = []
    
    for qa_pair in qa_pairs:
        # Only augment pairs that have questions
        if qa_pair.question is None:
            continue
        
        # Tokenize the question into words
        words = word_tokenize(qa_pair.question)
        
        # Only proceed if there are at least 2 words (needed for swapping)
        if len(words) < 2:
            continue
            
        # Generate multiple augmented versions of the same QA pair
        for _ in range(augmentation_factor):
            # Create a copy of the original words for this augmentation
            augmented_words = words.copy()
            
            # Perform n random swaps
            for _ in range(num_swaps):
                # Pick two different random positions
                pos1, pos2 = random.sample(range(len(augmented_words)), 2)
                
                # Swap the words
                augmented_words[pos1], augmented_words[pos2] = augmented_words[pos2], augmented_words[pos1]
            
            # Reconstruct the augmented question
            augmented_question = " ".join(augmented_words)
            
            # Create a new QA pair with the augmented question
            new_pair = QAPair(
                question=augmented_question,
                answer=qa_pair.answer,
                context=qa_pair.context
            )
            
            augmented_pairs.append(new_pair)
    
    return augmented_pairs