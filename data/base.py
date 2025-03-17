from .pipelines import init_setup, data_expansion, build_knowledge_graph, combination_segmentation, filter_data, data_augmentation

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from copy import deepcopy

nltk.download('punkt_tab')
nltk.download("stopwords")

class QAPair:
    _id_counter = 0

    def __init__(self, context: str = None, question: str = None, answer: str = None):
        self.id = QAPair._id_counter
        QAPair._id_counter += 1
        self.context = context
        self.question = question
        self.answer = answer
        self.links = []

        self.context_keywords = self.extract_keywords(context)
        self.question_keywords = self.extract_keywords(question)
        self.answer_keywords = self.extract_keywords(answer)

    def extract_keywords(self, text):
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

    def add_link(self, target_id):
        if target_id not in self.links:
            self.links.append(target_id)

    def classify(self):
        has_context = self.context is not None
        has_question = self.question is not None
        has_answer = self.answer is not None

        if has_context and not has_question and not has_answer:  # 1 0 0
            return "Document"
        elif not has_context and has_question and not has_answer:  # 0 1 0
            return "Unanswered Question"
        elif not has_context and not has_question and has_answer:  # 0 0 1
            return "Ground Truth"
        elif has_context and has_question and not has_answer:  # 1 1 0
            return "Context + Unanswered Question"
        elif has_context and not has_question and has_answer:  # 1 0 1
            return "Context + Ground Truth"
        elif not has_context and has_question and has_answer:  # 0 1 1
            return "QA Pair"
        elif has_context and has_question and has_answer:  # 1 1 1
            return "Full QA"
        else:
            return "Unknown"

    def classify_id(self):
        return {
            "Document": 1,
            "Unanswered Question": 2,
            "Ground Truth": 4,
            "Context + Unanswered Question": 3,
            "Context + Ground Truth": 5,
            "QA Pair": 6,
            "Full QA": 7,
            "Unknown": -1
        }[self.classify()]

    def __repr__(self):
        return (f"QAPair(id={self.id}, type={self.classify_id()}, context={self.context}, "
                f"question={self.question}, answer={self.answer}, links={self.links}, "
                f"context_keywords={self.context_keywords}, "
                f"question_keywords={self.question_keywords}, "
                f"answer_keywords={self.answer_keywords})")


class QADataset:
    def __init__(self):
        self.data = {}

    def get(self, id):
        if id in self.data:
            return self.data.get(id)
        return None

    def add(self, qa_pair: QAPair):
        new_id = len(self.data) 
        if new_id in self.data:
            raise ValueError("ID already exists in the dataset.")
        new_qa_pair = deepcopy(qa_pair)  
        new_qa_pair.id = new_id
        self.data[new_id] = new_qa_pair

    def add_link(self, source_id, target_id):
        if source_id in self.data and target_id in self.data:
            self.data[source_id].add_link(target_id)

    def filter_by_type(self, qa_type: str):
        return [qa for qa in self.data if qa.classify() == qa_type]

    def get_statistics(self):
        stats = {}
        for qa in self.data:
            qa_type = qa.classify()
            stats[qa_type] = stats.get(qa_type, 0) + 1
        return stats
    
    def run_pipeline(self, config):
        print("Running Init Setup...")
        self = init_setup(self, config.get("init_setup", {}))

        print("Running Data Expansion...")
        self = data_expansion(self, config.get("data_expansion", {}))

        print("Building Knowledge Graph...")
        self.data = build_knowledge_graph(self.data, config.get("build_knowledge_graph", {}))  

        print("Running Combination & Segmentation...")
        self = combination_segmentation(self, config.get("combination_segmentation", {}))

        print("Running Filter...")
        self = filter_data(self, config.get("filter", {}))

        print("Running Data Augmentation...")
        self = data_augmentation(self, config.get("data_augmentation", {}))

        print("Pipeline completed.")

    def __iter__(self):
        return iter(self.data.values())

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"QADataset(size={len(self.data)})"
