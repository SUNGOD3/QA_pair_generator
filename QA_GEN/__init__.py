from .base import QAPair, QADataset
from .pipelines import Pipeline
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download("stopwords")

__all__ = [
    "QAPair",
    "QADataset",
    "Pipeline",
    "MethodRegistry",
    "ProcessingMethod",
    "example_method"
]
