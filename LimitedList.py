
class LimitedList:

    def __init__(self, limit):
        self.limit = limit
        self.list = [None for index in range(0, self.limit)]
        self.current_index = 0

    def add(self, value):
        self.check_index()
        self.list[self.current_index] = value
        self.current_index += 1

    def check_index(self):
        if self.current_index == self.limit:
            self.current_index = 0
            self.list = [None for index in range(0, self.limit)]

    def check_value(self, value):
        if self.list.count(value) == self.limit:
            return True
        else:
            return False