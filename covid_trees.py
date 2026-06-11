class CircularQueue:
    """Fixed-size circular queue (FIFO) implementation."""

    def __init__(self, max_size):
        self.size = max_size
        self.queue = [None] * self.size
        self.front = 0
        self.rear = 0
        self.count = 0

    def enqueue(self, value):
        if self.is_full():
            print("Queue is full!")
            return
        self.queue[self.rear] = value
        self.rear = (self.rear + 1) % self.size
        self.count += 1

    def dequeue(self):
        if self.is_empty():
            print("Queue is empty!")
            return None
        value = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.size
        self.count -= 1
        return value

    def is_empty(self):
        return self.count == 0

    def is_full(self):
        return self.count == self.size

    def display(self):
        if self.is_empty():
            print("Queue is empty!")
            return
        idx = self.front
        elements = []
        for _ in range(self.count):
            elements.append(self.queue[idx])
            idx = (idx + 1) % self.size
        print("Queue elements:", elements)


# --- Test ---
q = CircularQueue(5)
for v in [10, 20, 30, 40, 50]:
    q.enqueue(v)
q.display()

q.dequeue()
q.dequeue()
q.display()

q.enqueue(60)
q.enqueue(70)
q.display()
