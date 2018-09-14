# coding: utf-8

from io import BytesIO
import os
from os.path import dirname
import pathlib
import logging
from sklearn.externals import joblib
from urllib.parse import urlparse
from triage.component.results_schema import TestEvaluation, TrainEvaluation, \
    TestPrediction, TrainPrediction

import pandas as pd
import s3fs
import yaml


class Store(object):
    def __init__(self, *pathparts):
        self.pathparts = pathparts

    @classmethod
    def factory(self, *pathparts):
        path_parsed = urlparse(pathparts[0])
        scheme = path_parsed.scheme

        if scheme in ('', 'file'):
            return FSStore(*pathparts)
        elif scheme == 's3':
            return S3Store(*pathparts)
        elif scheme == 'memory':
            return MemoryStore(*pathparts)
        else:
            raise ValueError('Unable to infer correct Store from project path')

    def __str__(self):
        return f"{self.__class__.__name__}(path={self.path})"

    def __repr__(self):
        return str(self)

    def exists(self):
        raise NotImplementedError

    def load(self):
        with self.open('rb') as fd:
            return fd.read()

    def write(self, bytestream):
        with self.open('wb') as fd:
            fd.write(bytestream)

    def open(self, *args, **kwargs):
        raise NotImplementedError


class S3Store(Store):
    def __init__(self, *pathparts):
        self.path = str(pathlib.PurePosixPath(pathparts[0].replace('s3://', ''), *pathparts[1:]))

    def exists(self):
        s3 = s3fs.S3FileSystem()
        return s3.exists(self.path)

    def delete(self):
        s3 = s3fs.S3FileSystem()
        s3.rm(self.path)

    def open(self, *args, **kwargs):
        s3 = s3fs.S3FileSystem()
        return s3.open(self.path, *args, **kwargs)


class FSStore(Store):
    def __init__(self, *pathparts):
        self.path = pathlib.Path(*pathparts)
        os.makedirs(dirname(self.path), exist_ok=True)

    def exists(self):
        return os.path.isfile(self.path)

    def delete(self):
        os.remove(self.path)

    def open(self, *args, **kwargs):
        return open(self.path, *args, **kwargs)


class MemoryStore(Store):
    store = None

    def __init__(self, *pathparts):
        pass

    def exists(self):
        return self.store is not None

    def delete(self):
        self.store = None

    def write(self, bytestream):
        self.store = bytestream

    def load(self):
        return self.store

    def open(self, *args, **kwargs):
        raise ValueError('MemoryStore objects cannot be opened and closed like files'
                         'Use write/load methods instead.')


class ProjectStorage(object):
    def __init__(self, project_path):
        self.project_path = project_path
        self.storage_class = Store.factory(self.project_path).__class__

    def get_store(self, directories, leaf_filename):
       return self.storage_class(self.project_path, *directories, leaf_filename)
   
    def matrix_storage_engine(self, matrix_storage_class=None, matrix_directory=None):
       return MatrixStorageEngine(self, matrix_storage_class, matrix_directory)


class ModelStorageEngine(object):
    """Store arbitrary models in a given project storage using joblib"""
    def __init__(self, project_storage, model_directory='trained_models'):
        self.project_storage = project_storage
        self.directories = [model_directory]

    def write(self, obj, model_hash):
        with self._get_store(model_hash).open('wb') as fd:
            joblib.dump(obj, fd, compress=True)

    def load(self, model_hash):
        with self._get_store(model_hash).open('rb') as fd:
            return joblib.load(fd)

    def exists(self, model_hash):
        return self._get_store(model_hash).exists()

    def delete(self, model_hash):
        return self._get_store(model_hash).delete()

    def _get_store(self, model_hash):
        return self.project_storage.get_store(self.directories, model_hash)


class MatrixStorageEngine(object):
    def __init__(self, project_storage, matrix_storage_class=None, matrix_directory=None):
        self.project_storage = project_storage
        self.matrix_storage_class = matrix_storage_class or CSVMatrixStore
        self.directories = [matrix_directory or 'matrices']

    def get_store(self, matrix_uuid):
        return self.matrix_storage_class(self.project_storage, self.directories, matrix_uuid)


class MatrixStore(object):
    _labels = None

    def __init__(self, project_storage, directories, matrix_uuid, matrix=None, metadata=None):
        self.matrix_uuid = matrix_uuid
        self.matrix_base_store = project_storage.get_store(directories, f'{matrix_uuid}.{self.suffix}')
        self.metadata_base_store = project_storage.get_store(directories, f'{matrix_uuid}.yaml')

        self.matrix = matrix
        self.metadata = metadata

    @property
    def matrix(self):
        if self.__matrix is None:
            self.__matrix = self._load()
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix):
        self.__matrix = matrix

    @property
    def metadata(self):
        if self.__metadata is None:
            self.__metadata = self.load_metadata()
        return self.__metadata

    @metadata.setter
    def metadata(self, metadata):
        self.__metadata = metadata

    @property
    def head_of_matrix(self):
        return self.matrix.head(1)

    @property
    def exists(self):
        return self.matrix_base_store.exists() and self.metadata_base_store.exists()

    @property
    def empty(self):
        if not self.matrix_base_store.exists():
            return True
        else:
            head_of_matrix = self.head_of_matrix
            return head_of_matrix.empty

    def columns(self, include_label=False):
        head_of_matrix = self.head_of_matrix
        columns = head_of_matrix.columns.tolist()
        if include_label:
            return columns
        else:
            return [
                col for col in columns
                if col != self.metadata['label_name']
            ]

    def labels(self):
        if self._labels is not None:
            logging.debug('using stored labels')
            return self._labels
        else:
            logging.debug('popping labels from matrix')
            self._labels = self.matrix.pop(self.metadata['label_name'])
            return self._labels

    @property
    def uuid(self):
        return self.matrix_uuid

    @property
    def as_of_dates(self):
        if 'as_of_date' in self.matrix.index.names:
            return sorted(list(set([as_of_date for entity_id, as_of_date in self.matrix.index])))
        else:
            return [self.metadata['end_time']]

    @property
    def num_entities(self):
        if self.matrix.index.names == ['entity_id']:
            return len(self.matrix.index.values)
        elif 'entity_id' in self.matrix.index.names:
            return len(self.matrix.index.levels[self.matrix.index.names.index('entity_id')])

    @property
    def matrix_type(self):
        if self.metadata['matrix_type'] == 'train':
            return TrainMatrixType
        elif self.metadata['matrix_type'] == 'test':
            return TestMatrixType
        else:
            raise Exception('''matrix metadata for matrix {} must contain 'matrix_type'
             = "train" or "test" '''.format(self.uuid))

    def matrix_with_sorted_columns(self, columns):
        columnset = set(self.columns())
        desired_columnset = set(columns)
        if columnset == desired_columnset:
            if self.columns() != columns:
                logging.warning('Column orders not the same, re-ordering')
            return self.matrix[columns]
        else:
            if columnset.issuperset(desired_columnset):
                raise ValueError('''
                    Columnset is superset of desired columnset. Extra items: %s
                ''', columnset - desired_columnset)
            elif columnset.issubset(desired_columnset):
                raise ValueError('''
                    Columnset is subset of desired columnset. Extra items: %s
                ''', desired_columnset - columnset)
            else:
                raise ValueError('''
                    Columnset and desired columnset mismatch. Unique items: %s
                ''', columnset ^ desired_columnset)

    def load_metadata(self):
        with self.metadata_base_store.open('rb') as fd:
            return yaml.load(fd)

    def save(self):
        raise NotImplementedError

    def __getstate__(self):
        # when we serialize (say, for multiprocessing),
        # we don't want the cached members to show up
        # as they can be enormous
        self.matrix = None
        self._labels = None
        self.metadata = None
        return self.__dict__.copy()


class HDFMatrixStore(MatrixStore):
    suffix = 'h5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.matrix_base_store, S3Store):
            raise ValueError('HDFMatrixStore cannot be used with S3')

    @property
    def head_of_matrix(self):
        try:
            head_of_matrix = pd.read_hdf(self.matrix_base_store.path, start=0, stop=1)
            # Is the index already in place?
            if head_of_matrix.index.names != self.metadata['indices']:
                head_of_matrix.set_index(self.metadata['indices'], inplace=True)
        except pd.errors.EmptyDataError:
            head_of_matrix = None

        return head_of_matrix

    def _load(self):
        matrix = pd.read_hdf(self.matrix_base_store.path)

        # Is the index already in place?
        if matrix.index.names != self.metadata['indices']:
            matrix.set_index(self.metadata['indices'], inplace=True)

        return matrix

    def save(self):
        hdf = pd.HDFStore(self.matrix_base_store.path,
                          mode='w',
                          complevel=4,
                          complib="zlib",
                          format='table')
        hdf.put(self.matrix_uuid, self.matrix.apply(pd.to_numeric), data_columns=True)
        hdf.close()

        with self.metadata_base_store.open('wb') as fd:
            yaml.dump(self.metadata, fd, encoding='utf-8')


class CSVMatrixStore(MatrixStore):
    suffix = 'csv'

    @property
    def head_of_matrix(self):
        try:
            with self.matrix_base_store.open('rb') as fd:
                head_of_matrix = pd.read_csv(fd, nrows=1)
                head_of_matrix.set_index(self.metadata['indices'], inplace=True)
        except FileNotFoundError as fnfe:
            logging.exception(f"Matrix isn't there: {fnfe}")
            logging.exception("Returning Empty data frame")
            head_of_matrix = pd.DataFrame()

        return head_of_matrix

    def _load(self):
        with self.matrix_base_store.open('rb') as fd:
            matrix = pd.read_csv(fd)

        matrix.set_index(self.metadata['indices'], inplace=True)

        return matrix

    def save(self):
        self.matrix_base_store.write(self.matrix.to_csv(None).encode('utf-8'))
        with self.metadata_base_store.open('wb') as fd:
            yaml.dump(self.metadata, fd, encoding='utf-8')


class TestMatrixType(object):
    string_name = 'test'
    evaluation_obj = TestEvaluation
    prediction_obj = TestPrediction
    is_test = True


class TrainMatrixType(object):
    string_name = 'train'
    evaluation_obj = TrainEvaluation
    prediction_obj = TrainPrediction
    is_test = False
