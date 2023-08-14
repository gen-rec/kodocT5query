import os.path
import pickle

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from src.utils import TokenizeCollate


class DocT5QueryDataset(Dataset):
    def __init__(
        self,
        qid_to_docid: list[tuple[str, str]],
        collections: dict[str, str],
        queries: dict[str, str],
        return_labels: bool = True,
    ):
        """
        Dataset for docT5query

        Args:
            qid_to_docid (list[tuple[str, str]]): List of tuples of query id and document id
            collections (dict[str, str]): Dictionary of document id to document
            queries (dict[str, str]): Dictionary of query id to query
            return_labels (bool, optional): Whether to return labels. Defaults to True.
        """
        self.qid_to_docid = qid_to_docid
        self.collections = collections
        self.queries = queries
        self.return_labels = return_labels

    def __len__(self):
        return len(self.qid_to_docid)

    def __getitem__(self, idx: int) -> tuple[str, str] | str:
        """
        Get a document and query pair, or just a document

        Args:
            idx (int): Index of the document and query pair

        Returns:
            tuple[str, str] | str: Document and query pair, or just a document

        """
        qid, doc_id = self.qid_to_docid[idx]
        query = self.queries[qid]
        doc = self.collections[doc_id]

        if self.return_labels:
            return doc, query
        else:
            return doc


class DocT5QueryDataModule(LightningDataModule):
    COLLECTIONS_FILE_NAME = "collection.tsv"
    QUERIES_FILE_NAME = "questions.tsv"
    QRELS_FILE_NAME = "qrels.tsv"
    TRAIN_QIDS_FILE_NAME = "train_qids.pkl"
    VALID_QIDS_FILE_NAME = "valid_qids.pkl"
    TEST_QIDS_FILE_NAME = "test_qids.pkl"

    def __init__(
        self,
        dataset_paths: list[str],
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        batch_size: int,
        num_workers: int = 4,
    ):
        """
        DataModule for the DocT5QueryModule

        Args:
            dataset_paths (list[str]): List of paths to the dataset. Each path contains the collection, queries, and
                qrels files, as well as the train/valid/test qids files

                File names are:
                    * `COLLECTIONS_FILE_NAME = "collection.tsv"`
                    * `QUERIES_FILE_NAME = "questions.tsv"`
                    * `QRELS_FILE_NAME = "qrels.tsv"`
                    * `TRAIN_QIDS_FILE_NAME = "train_qids.pkl"`
                    * `VALID_QIDS_FILE_NAME = "valid_qids.pkl"`
                    * `TEST_QIDS_FILE_NAME = "test_qids.pkl"`

            tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to use
            batch_size (int): Batch size
            num_workers (int, optional): Number of workers for the DataLoader. Defaults to 4.
        """
        super().__init__()

        self.collections = {}  # doc_id -> doc
        self.queries = {}  # query_id -> query
        self.qrels = {}  # query_id -> doc_id

        self.train_qids = set()  # query_id
        self.valid_qids = set()  # query_id
        self.test_qids = set()  # query_id

        for dataset_path in dataset_paths:
            self.collections.update(
                self._load_map(
                    os.path.join(dataset_path, self.COLLECTIONS_FILE_NAME), "||"
                )
            )
            self.queries.update(
                self._load_map(os.path.join(dataset_path, self.QUERIES_FILE_NAME), "\t")
            )
            self.qrels.update(
                self._load_map(os.path.join(dataset_path, self.QRELS_FILE_NAME), "\t")
            )

            self.train_qids.update(
                pickle.load(
                    open(os.path.join(dataset_path, self.TRAIN_QIDS_FILE_NAME), "rb")
                )
            )
            self.valid_qids.update(
                pickle.load(
                    open(os.path.join(dataset_path, self.VALID_QIDS_FILE_NAME), "rb")
                )
            )
            self.test_qids.update(
                pickle.load(
                    open(os.path.join(dataset_path, self.TEST_QIDS_FILE_NAME), "rb")
                )
            )

        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.qid_to_docid = [
            (qid, self.qrels[qid]) for qid in self.queries
        ]  # [(qid, docid), ...]

        # Split dataset
        self.train = []
        self.val = []
        self.test = []

        for qid, docid in self.qid_to_docid:
            if qid in self.train_qids:
                self.train.append((qid, docid))
            elif qid in self.valid_qids:
                self.val.append((qid, docid))
            elif qid in self.test_qids:
                self.test.append((qid, docid))
            else:
                raise ValueError(
                    f"Query id {qid} not found in neither train, val, nor test"
                )

    def train_dataloader(self):
        dataset = DocT5QueryDataset(self.train, self.collections, self.queries)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=TokenizeCollate(self.tokenizer),
        )

    def val_dataloader(self):
        dataset = DocT5QueryDataset(self.val, self.collections, self.queries)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TokenizeCollate(self.tokenizer),
        )

    def test_dataloader(self):
        dataset = DocT5QueryDataset(self.test, self.collections, self.queries)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TokenizeCollate(self.tokenizer),
        )

    def predict_dataloader(self):
        dataset = DocT5QueryDataset(
            self.test, self.collections, self.queries, return_labels=False
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=TokenizeCollate(self.tokenizer, return_labels=False),
        )

    @staticmethod
    def _load_map(path: str, delimiter: str) -> dict[str, str]:
        """
        Load a map from a file

        Args:
            path (str): Path to the file
            delimiter (str): Delimiter to use for splitting the lines

        Examples:
            >>> _load_map("path/to/file", "\\t")
            {"key1": "value1", "key2": "value2"}

        Returns:
            dict[str, str]: Map from the file
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_collections = f.readlines()

        raw_collections = [x.strip().split(delimiter) for x in raw_collections]

        collections = {doc_id: text for doc_id, text in raw_collections}

        return collections
