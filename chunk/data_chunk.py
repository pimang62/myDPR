from abc import ABC, abstractmethod
from chunk.base import Chunker

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from unstructured.cleaners.core import clean_bullets, clean_extra_whitespace, clean_dashes, group_broken_paragraphs 
import re

import os
root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) 

def preprocess(text: str):
    """
    Remove duplicated white spaces from text

    example.
    '       ' -> ' '
    """
    text = text.replace('\n', ' ')
    return re.sub('\s+', ' ', text).strip()

# if you want to split Q & A
def split_contents(doc: str) -> tuple:  
    """Split question and content by newline"""
    question, *content = doc.split('\n')
    return preprocess(question), preprocess(' '.join(content))


class BaeminChunker(Chunker):
    """Chunk baemin.txt data"""
    def __init__(self, fname):
        super().__init__(fname)
        self.doc = None

    def load(self):
        loader = TextLoader(self.fname)  # Use self.fname
        self.doc = loader.load()
        self._is_loaded = True  
        print(f"Number of documents: {len(self.doc)}")

    def chunk(self, chunk_size, chunk_overlap):
        """
        Returns:
            texts (List[str]): list of texts
        """
        if not self._is_loaded:  # Check if data is loaded
            self.load() 
         
        # baemin.txt is delimited by "\n\n"
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"],
                                                       chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
        
        texts = [preprocess((docs.page_content)) for docs in text_splitter.split_documents(self.doc)]  # text_splitter.split_documents(self.doc) : [{Document: ...}, {}, ...]
        
        return texts


class TXTChunker(Chunker):
    """Chunk *.txt data"""
    def __init__(self, fname):
        super().__init__(fname)
        self.doc = None

    def load(self):
        loader = TextLoader(self.fname)  # Use self.fname
        self.doc = loader.load()
        self._is_loaded = True  
        print(f"Number of documents: {len(self.doc)}")

    def chunk(self, chunk_size, chunk_overlap):
        """
        Returns:
            texts (List[str]): list of texts
        """
        if not self._is_loaded:  # Check if data is loaded
            self.load() 
         
        # baemin.txt is delimited by "\n\n"
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"],
                                                       chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
        
        texts = [preprocess((docs.page_content)) for docs in text_splitter.split_documents(self.doc)]  # text_splitter.split_documents(self.doc) : [{Document: ...}, {}, ...]
        
        return texts
        

class PDFChunker(Chunker):
    """Chunk *.pdf data"""
    def __init__(self, fname):
        super().__init__(fname)
        self.doc = None

    def load(self):
        loader = UnstructuredFileLoader(self.fname, mode="single",
                                        # post_processors=[clean_bullets, 
                                        #                  clean_extra_whitespace, 
                                        #                  clean_dashes, 
                                        #                  group_broken_paragraphs]
                                        )
        self.doc = loader.load()
        self._is_loaded = True
        print(f"Number of documents: {len(self.doc)}")

    def chunk(self, chunk_size, chunk_overlap):
        """
        Returns:
            texts (List[str]): list of texts
        """
        if not self._is_loaded:  # Check if data is loaded
            self.load() 
        
        text_splitter = RecursiveCharacterTextSplitter(separators=[""],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        texts = [preprocess((docs.page_content)) for docs in text_splitter.split_documents(self.doc)]

        return texts


class DocxChunker(Chunker):
    """Chunk *.docx data"""
    def __init__(self, fname):
        super().__init__(fname)
        self.doc = None

    def load(self):
        loader = UnstructuredFileLoader(self.fname, mode="single",
                                        # post_processors=[clean_bullets, 
                                        #                  clean_extra_whitespace, 
                                        #                  clean_dashes, 
                                        #                  group_broken_paragraphs]
                                        )
        self.doc = loader.load()
        self._is_loaded = True
        print(f"Number of documents: {len(self.doc)}")

    def chunk(self, chunk_size, chunk_overlap):
        """
        Returns:
            texts (List[str]): list of texts
        """
        if not self._is_loaded:  # Check if data is loaded
            self.load() 
        
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"],
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        texts = [preprocess((docs.page_content)) for docs in text_splitter.split_documents(self.doc)]

        return texts

## chunk factory
class ChunkerFactory:
    """Factory for chunking"""
    def __init__(self, fname):
        self.fname = fname
        self.data_type = fname.split(".")[-1]

    def create_chunker(self):
        if "txt" == self.data_type:
            return TXTChunker(self.fname)
        elif "pdf" == self.data_type:
            return PDFChunker(self.fname)
        elif "docx" == self.data_type:
            return DocxChunker(self.fname)
        elif "tsv" == self.data_type:  # baemin, baemin_expert
            return BaeminChunker(self.fname)
        else:
            raise Exception("Not supported data type.")