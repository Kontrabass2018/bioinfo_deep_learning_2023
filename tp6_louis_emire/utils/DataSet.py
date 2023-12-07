import h5py
import numpy as np

class DataSet:
    def __init__(self, data_set_path: str) -> None:
        super().__init__()
        self.data_set_path = data_set_path

    def _load_data(self):
        return h5py.File(self.data_set_path, "r")

    def _parse_file_data(self, column, dtype: type):
        return np.array(column, dtype=dtype)
    
    def get_data(self, column_name: str, type: type):
        """
        Class orchestrator
        """
        fi = self._load_data()
        
        return self._parse_file_data(fi[column_name], type)