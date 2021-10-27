"""
Abstract class for datasets for continual learning. Defines attributes and functions
that any dataset used for continual learning should have.
"""

class ContinualDataset(BaseDataset):
    """ Abstract class for continual learning datasets. """

    def __init__(self) -> None:
        """
        Init function for ContinualDatset. Initializes all required members, but to null
        values. Any subclass of ContinualDataset should initialize these values AFTER
        calling ContinualDataset.__init__(). This function mostly just exists to
        document the required members that must be populated by any subclass.
        """

        super(ContinualDataset, self).__init__()

        self.num_tasks = None
        self.current_task = None
        self.task_datasets = []

    def __len__(self):
        return len(self.task_datasets[self.current_task])

    def __getitem__(self, idx: int):
        """ Return the item with index `idx` from the dataset for the current task. """
        return self.task_datasets[self.current_task][idx]
