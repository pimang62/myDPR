from abc import ABC, abstractmethod

class Chunker(ABC):
    """
    Abstract class for data chunking
    Write your own chunker depending on the data structure.
    """
    def __init__(self, fname):
        self.fname = fname
        self._is_loaded = False  # State tracking

    @abstractmethod
    def load(self):
        """Load data from given file name"""
        pass

    @abstractmethod
    def chunk(self, chunk_size, chunk_overlap):
        """Chunk data with given chunk size and overlap"""
        pass

## load, chunk 부분은 추후에 데이터가 일반화되고 많아지면 클래스 분리될수도 있음
## - 통으로 된 문서를 쪼갤 수도 있고, (ex. 규정집)
## delimiter로 나뉘어진 한개의 문서일 수도 있고 (ex. 배민)
## 7번 데이터 처럼 여러개의 폴더에 문서가 정제된 형태로 나뉘어져있을수 있다 (ex. nia 데이터)
## TODO: Chunking based on character and tokenizer 