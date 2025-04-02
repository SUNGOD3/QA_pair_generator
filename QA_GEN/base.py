from typing import Dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from copy import deepcopy

nltk.download('punkt_tab')
nltk.download("stopwords")

class QAPair:

    def __init__(self, context: str = None, question: str = None, answer: str = None):
        self.id = None
        self.set_answer(answer)
        self.set_context(context)
        self.set_question(question)
        self.edges = []

    def extract_keywords(self, text):
        '''
        Use NLTK to extract the top 5 most common keywords from the text.
        '''
        if not text:
            return []
        words = word_tokenize(text)
        words = [w.lower() for w in words if w.isalnum()] # Remove punctuation
        stop_words = set(stopwords.words("english"))
        filtered_words = [w for w in words if w not in stop_words]

        # Count the frequency of each word
        keyword_counts = Counter(filtered_words)
        return [word for word, _ in keyword_counts.most_common(5)]
    
    def set_context(self, context):
        self.context = context
        self.context_keywords = self.extract_keywords(context)
    
    def set_question(self, question):
        self.question = question
        self.question_keywords = self.extract_keywords(question)
    
    def set_answer(self, answer):
        self.answer = answer
        self.answer_keywords = self.extract_keywords(answer)

    def add_edge(self, target_id):
        if target_id not in self.edges:
            self.edges.append(target_id)

    def classify(self):
        '''
        Classify the QA pair based on the presence of context, question, and answer.
        '''
        has_context = self.context is not None
        has_question = self.question is not None
        has_answer = self.answer is not None

        if has_context and not has_question and not has_answer:  # 1 0 0
            return "Document"
        elif not has_context and has_question and not has_answer:  # 0 1 0
            return "Question"
        elif not has_context and not has_question and has_answer:  # 0 0 1
            return "Ground Truth"
        elif has_context and has_question and not has_answer:  # 1 1 0
            return "Context + Question"
        elif has_context and not has_question and has_answer:  # 1 0 1
            return "Context + Ground Truth"
        elif not has_context and has_question and has_answer:  # 0 1 1
            return "QA Pair"
        elif has_context and has_question and has_answer:  # 1 1 1
            return "Complete QA"
        else:
            return "Unknown"

    def classify_id(self):
        return {
            "Document": 1,
            "Question": 2,
            "Ground Truth": 4,
            "Context + Question": 3,
            "Context + Ground Truth": 5,
            "QA Pair": 6,
            "Complete QA": 7,
            "Unknown": -1
        }[self.classify()]

    def __repr__(self):
        return (f"QAPair(id={self.id}, type={self.classify_id()}, context={self.context}, "
                f"question={self.question}, answer={self.answer}, edges={self.edges}, "
                f"context_keywords={self.context_keywords}, "
                f"question_keywords={self.question_keywords}, "
                f"answer_keywords={self.answer_keywords})")


class QADataset:
    def __init__(self, name: str = None, description: str = None, qa_pairs: list[QAPair] = None):
        self.name = name
        self.description = description
        self.data: dict[int, QAPair] = {}

        if qa_pairs:
            self._initialize_dataset(qa_pairs)

    def _initialize_dataset(self, qa_pairs: list[QAPair]):
        '''
        Reassigns the IDs of the QA pairs in the dataset.
        '''
        id_mapping = {} 
        new_data = {}
        for new_id, qa_pair in enumerate(qa_pairs, start=1):
            id_mapping[qa_pair.id] = new_id
            qa_pair.id = new_id
            new_data[new_id] = qa_pair
        for qa in new_data.values():
            qa.edges = [id_mapping[old_id] for old_id in qa.edges if old_id in id_mapping]

        self.data = new_data


    def get(self, id):
        if id in self.data:
            return self.data.get(id)
        return None

    def add(self, qa_pair: QAPair):
        new_id = len(self.data) + 1
        if new_id in self.data:
            raise ValueError("ID already exists in the dataset.")
        new_qa_pair = deepcopy(qa_pair)  
        new_qa_pair.id = new_id
        self.data[new_id] = new_qa_pair

    def add_edge(self, source_id, target_id):
        if source_id in self.data and target_id in self.data:
            self.data[source_id].add_edge(target_id)

    def filter_by_type(self, qa_type: str):
        return [qa for qa in self.data if qa.classify() == qa_type]

    def get_statistics(self):
        '''
        Return the count of each type of QA pair in the dataset.
        '''
        stats = {}
        for qa in self.data:
            qa_type = qa.classify()
            stats[qa_type] = stats.get(qa_type, 0) + 1
        return stats

    def __iter__(self):
        return iter(self.data.values())

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"QADataset(size={len(self.data)})"
    
    def copy(self):
        return QADataset(
            name=self.name,
            description=self.description,
            qa_pairs=list(self.data.values())
        )
