import pickle
import os


class BufferedPickleContainer:
    def __init__(self, path, flush_every=100):
        self.path = path
        self.flush_every = flush_every
        self.buffer = []

        # ensure file exists
        if not os.path.exists(path):
            open(path, "wb").close()

    def add(self, item):
        """Add an item to the container."""
        self.buffer.append(item)

        if len(self.buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        """Write buffered items to disk."""
        if not self.buffer:
            return

        with open(self.path, "ab") as f:
            for item in self.buffer:
                pickle.dump(item, f)

        self.buffer.clear()

    def close(self):
        """Flush remaining items."""
        self.flush()

    def __iter__(self):
        """Iterate through stored items."""
        with open(self.path, "rb") as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break

    def __len__(self):
        """Count items in file."""
        count = 0
        for _ in self:
            count += 1
        return count
