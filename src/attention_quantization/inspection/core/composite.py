from .collector import Collector


class CompositeCollector(Collector):
    """Coordinate collectors, including rollback on partial registration."""

    def __init__(self, children):
        """Retain ordered child collectors; no hooks are registered yet."""
        self.children = list(children)

    def start(self, publish):
        """Start children with one target/tensor callback; roll back on failure."""
        try:
            for child in self.children:
                child.start(publish)
        except BaseException:
            self.close()
            raise

    def close(self):
        """Release child resources in reverse registration order."""
        for child in reversed(self.children):
            child.close()
