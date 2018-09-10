"""Adaptive Sliding Window"""


from test_ADWIN.adwin_list import AdwinList
from math import log, sqrt, fabs


class Adwin:
    def __init__(self, delta=0.002, max_buckets=5, min_clock=32, min_length_window=8, min_length_sub_window=4):
        """
        :param delta: confidence value
        :param max_buckets: max number of buckets which have same number of original date in one row
        :param min_clock: min number of new data for starting to reduce window and detect change
        :param min_length_window: min window's length for starting to reduce window and detect change
        :param min_length_sub_window: min sub window's length for starting to reduce window and detect change
        """
        self.delta = delta
        self.max_number_of_buckets = max_buckets
        self.min_number_new_data = min_clock
        self.min_length_window = min_length_window
        self.min_length_sub_window = min_length_sub_window
        # time is used for the sake of comparing with min_number_new_data parameter
        self.time = 0
        # width of the window
        self.width = 0
        # sum of all instances in the window
        self.total = 0.0
        # incremental variance of all instances in a window
        self.var = 0.0
        # Increment the number of buckets up to the value of max_buckets
        self.bucket_number = 0
        self.last_bucket_row = 0  # defines the max number of merged
        self.list_buckets = AdwinList(self.max_number_of_buckets)

    def set_input(self, value):
        """
        add a new data instance and automatically detects a possible concept drift.
        :param value: new data value
        :return: true if there is a concept drift, otherwise false
        """
        self.time += 1
        # Insert the new element
        self.insert_element(value)
        # Reduce window
        return self.reduce_window()

    def insert_element(self, value):
        """
        Insert a new element by creating a new bucket for the head element of the list. variance and
        total value are updated incrementally. we must compressed buckets (merged) if the maximum number of
        buckets has been reached. cf Albert bifet et Gavalda : Learning from time changing data withAdaptive Win.
        :param value: newly arrived data value from the stream
        """
        self.width += 1
        # Insert the new bucket at the head of the bucket
        self.list_buckets.head.insert_bucket(value, 0)
        self.bucket_number += 1
        inc_var = 0
        if self.width > 1:
            inc_var = (self.width - 1) * pow(2, (value - self.total / (self.width - 1))) / self.width
        self.var += inc_var
        self.total += value
        # compress buckets if condition met.
        self.compress_buckets()

    def compress_buckets(self):
        """Merge two buckets, i.e create a new bucket of size equals to the sum of the sizes of
        those two buckets.
        """
        cursor_on_bucket = self.list_buckets.head
        i = 0
        while cursor_on_bucket is not None:
            # Find the number of buckets in a row
            k = cursor_on_bucket.bucket_size_row

            # Merge buckets
            if k == self.max_number_of_buckets + 1:
                next_node = cursor_on_bucket.next
                if next_node is None:
                    self.list_buckets.add_to_tail()
                    next_node = cursor_on_bucket.next
                    self.last_bucket_row += 1

                n1 = pow(2, i)
                n2 = pow(2, i)

                u1 = cursor_on_bucket.bucket_total[0] / n1
                u2 = cursor_on_bucket.bucket_total[1] / n2

                ext_var = n1 * n2 * (u1 - u2) * (u1 - u2) / (n1 + n2)

                # Creating and inserting a new bucket in the next list item
                new_bucket_total = cursor_on_bucket.bucket_total[0] + cursor_on_bucket.bucket_total[1]
                new_bucket_variance = cursor_on_bucket.bucket_variance[0] + cursor_on_bucket.bucket_variance[1] + ext_var
                next_node.insert_bucket(new_bucket_total, new_bucket_variance)
                self.bucket_number += 1

                # compress and hence, removes 2 buckets from the current list item
                cursor_on_bucket.compress_buckets_row(2)

                # stop if the the max number of buckets does not exceed for the next item list  
                if next_node.bucket_size_row <= self.max_number_of_buckets:
                    break
            else:
                break
            cursor_on_bucket = cursor_on_bucket.next
            i += 1

    def reduce_window(self):
        """
        reduce the window if there is a concept drift.
        :return: boolean: change occurred or not.
        """
        is_changed_occurred = False
        if self.time % self.min_number_new_data == 0 and self.width > self.min_length_window:
            is_shrink_width = True
            while is_shrink_width:
                is_shrink_width = False
                exit = False
                n0, n1 = 0, self.width
                u0, u1 = 0, self.total

                # start building sub windows
                cursor_on_bucket = self.list_buckets.tail
                i = self.last_bucket_row
                while (not exit) and (cursor_on_bucket is not None):
                    for k in range(cursor_on_bucket.bucket_size_row):
                        # In case of n1 equals 0
                        if i == 0 and k == cursor_on_bucket.bucket_size_row - 1:
                            exit = True
                            break

                        # One subwindow enlarges and the other one shrinks
                        n0 += pow(2, i)
                        n1 -= pow(2, i)
                        u0 += cursor_on_bucket.bucket_total[k]
                        u1 -= cursor_on_bucket.bucket_total[k]
                        diff_value = (u0 / n0) - (u1 / n1)

                        # remove old entries iff there is a concept drift and the minimum sub window length is matching
                        if n0 > self.min_length_sub_window + 1 and n1 > self.min_length_sub_window + 1 and \
                                self.reduce_expression(n0, n1, diff_value):
                            is_shrink_width, is_changed_occurred = True, True
                            if self.width > 0:
                                n0 -= self.delete_element()
                                exit = True
                                break
                    cursor_on_bucket = cursor_on_bucket.previous
                    i -= 1
        return is_changed_occurred

    def reduce_expression(self, n0, n1, diff_value):
        """
        Calculate epsilon_cut value and check if difference between the true mean value and the expected value of the two sub windows is greater than
        it.
        :param n0: total number of instances in sub window 0
        :param n1: total number of instances in sub window 1
        :param diff_value: difference of mean values of both sub windows
        :return: true if difference of mean values is higher than epsilon_cut
        """
        # harmonic mean of n0 and n1 (originally: 1 / (1/n0 + 1/n1))
        m = 1 / (n0 - self.min_length_sub_window + 1) + 1 / (n1 - self.min_length_sub_window + 1)
        d = log(2 * log(self.width) / self.delta)
        var = self.var / self.width
        epsilon_cut = sqrt(2 * m * var * d) + 2 / 3 * m * d
        return fabs(diff_value) > epsilon_cut

    def delete_element(self):
        """
        Remove a bucket from tail of window
        :return: Number of elements to be deleted
        """
        # last list item (the oldest bucket) with the oldest entry at first internal array position
        node = self.list_buckets.tail
        deleted_number = pow(2, self.last_bucket_row)
        self.width -= deleted_number
        self.total -= node.bucket_total[0]
        deleted_element_mean = node.bucket_total[0] / deleted_number

        incremental_variance = node.bucket_variance[0] + deleted_number * self.width * \
                                                         pow(2, (deleted_element_mean - self.total / self.width)) / \
                                                         (deleted_number + self.width)
        self.var -= incremental_variance

        node.compress_buckets_row(1)
        self.bucket_number -= 1
        # remove it from the tail, if the bucket becomes empty after compression and removal of an instance.
        if node.bucket_size_row == 0:
            self.list_buckets.remove_from_tail()
            self.last_bucket_row -= 1
        return deleted_number
