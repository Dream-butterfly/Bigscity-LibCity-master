from torch.utils.data import Dataset


class ListDataset(Dataset):
    def __init__(self, data):
        """
        data: 必须是一个 list
        """
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ArrayTupleDataset(Dataset):
    def __init__(self, *arrays):
        if not arrays:
            raise ValueError("ArrayTupleDataset requires at least one array")
        size = len(arrays[0])
        for array in arrays[1:]:
            if len(array) != size:
                raise ValueError("All arrays in ArrayTupleDataset must have the same length")
        self.arrays = arrays

    def __getitem__(self, index):
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return len(self.arrays[0])
