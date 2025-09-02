from typing import List
from QA_GEN.base import QAPair
from QA_GEN.method_register import Method
import random
from nltk.tokenize import word_tokenize

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