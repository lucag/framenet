"""FrameNet types
"""

from typing import NamedTuple, Sequence


class Node:
    def __init__(self, uid: int, name: str):
        self.uid, self.name = uid, name


class Edge:
    def __init__(self, node1: Node, node2: Node):
        self.edge = node1, node2


class SemType(Node):
    pass


class TypedNode(Node):
    def __init__(self, uid: int, name: str, sem_type: SemType =None):
        super(TypedNode).__init__(uid, name, sem_type)

    @property
    def sem_type(self):
        return self.sem_type


class FrameElement(TypedNode):
    pass


class Frame(TypedNode):
    def __init__(self, uid: int, name: str, elements: Sequence[FrameElement], sem_type=None):
        super(Frame).__init__(uid, name, sem_type)
        self.elements = elements


class Relation(Edge):
    def __init__(cls, uid, kind, node1, node2) -> None:
        self = super(Relation).__init__(node1, node2)
        self.kind = kind


class LexicalUnit(TypedNode):
    pass


class Lexeme: pass


class Lemma: pass


class WordForm: pass

