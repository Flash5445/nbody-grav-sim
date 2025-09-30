class Node:
    def __init__(self, m, p, v):
        self.m = m
        self.p = p
        self.v = v
        children = []
        

    def add_child(self, node):
        self.children.append(node)
        return None

    