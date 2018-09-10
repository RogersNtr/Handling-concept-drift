import numpy as np


class AdwinListItem:
    """
    A list item contains a list (array) of buckets limited to a maximum of buckets.
    Each item has a connection to a previous and next list item.
    """

    def __init__(self, max_buckets=5, next=None, previous=None):
        self.M = max_buckets
        # current number of buckets in this list item
        self.bucket_size_row = 0

        self.next = next
        # add the 'previous' connection of the following list item to this item
        if next is not None:
            next.previous = self

        self.previous = previous
        # next connection of the previous list item add to this item
        if previous is not None:
            previous.next = self

        self.bucket_total = np.zeros(self.M + 1)
        self.bucket_variance = np.zeros(self.M + 1)

    def insert_bucket(self, value, variance):
        """
        Insert a new bucket at the end of the array.
        """
        self.bucket_total[self.bucket_size_row] = value
        self.bucket_variance[self.bucket_size_row] = variance
        self.bucket_size_row += 1

    def compress_buckets_row(self, nber_to_delete):
        """
        Remove the  bucket number 'nber_to_delete' .
        """
        delete_index = self.M - nber_to_delete + 1
        self.bucket_total[:delete_index] = self.bucket_total[nber_to_delete:]
        self.bucket_total[delete_index:] = np.zeros(nber_to_delete)

        self.bucket_variance[:delete_index] = self.bucket_variance[nber_to_delete:]
        self.bucket_variance[delete_index:] = np.zeros(nber_to_delete)

        self.bucket_size_row -= nber_to_delete
