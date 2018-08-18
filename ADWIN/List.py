
from ADWIN.ListNode import *
class AdwinList:
    def __init__(self, M):
        """

        :param M: M controls the amount of memory used and the closeness of the check
        point.
        """
        self.M = M
        self.count = 0
        self.head = None
        self.tail = None
        self.add_to_head()

    def add_to_head(self):
        temp = ListNode(self.M)
        if self.head is not None:
            temp.next = self.head
            self.head.prev = temp
        self.head = temp
        if self.tail is None:
            self.tail = self.head
        self.count += 1

    def add_to_tail(self):
        temp = ListNode(self.M)
        if self.tail is not None:
            temp.prev = self.tail
            self.tail.next = temp
        self.tail = temp
        if self.head is None:
            self.head = self.tail
        self.count += 1

    def remove_from_head(self):
        temp = self.head
        self.head = self.head.next
        if self.head is not None:
            self.head.prev = None
        else:
            self.tail = None

        self.count -= 1
        del temp

    def remove_from_tail(self):
        temp = self.tail
        self.tail = self.tail.prev
        if self.tail is not None:
            self.tail.next = None
        else:
            self.head = None

        self.count -= 1
        del temp
