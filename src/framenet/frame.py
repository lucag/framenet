"""@author: <seantrott@icsi.berkeley.edu>

This module defines the Frame object, as well as the associated SemType class, FrameElement class, and FrameBuilder.

Other associated Frame classes are defined in frame_relation and lexical_units.
"""

import xml.etree.ElementTree as et

from framenet.lexical_unit  import ShallowLU
from pprint                 import pformat


class Node(object):
    """ Simple Node object, has parents and children. """

    def __init__(self, parents, children):
        self.parents = parents
        self.children = children


class Frame(Node):
    """ Represents a single FrameNet frame. Includes:
    -Frame Name
    -Frame elements 
    -Frame lexical units 
    -Frame relations 
    -Frame parents 
    -Frame children 
    -Frame definition (text and XML)
    -Frame ID
    """

    def __init__(self, name, elements, lexicalUnits, relations, parents, children, definition, xml_def, ID):
        Node.__init__(self, parents=parents, children=children)
        self.name = name
        self.elements = elements
        self.lexicalUnits = lexicalUnits
        self.relations = relations
        self.fe_relations = []
        self.xml_definition = xml_def
        self.definition = definition
        self.ID = ID
        self.individual_valences = []
        self.group_realizations = []
        self.fe_realizations = []
        self.annotations = []

    def is_related(self, frame):
        """ Checks if a string 'frame' is related to SELF. """
        for relation in self.relations:
            for related in relation.related_frames:
                if related.name == frame:
                    return relation.relation_type
        return False

    def add_valences(self, valences):
        self.individual_valences += valences

    def add_group_realizations(self, res):
        self.group_realizations += res

    def add_fe_realizations(self, res):
        self.fe_realizations += res

    def add_annotations(self, anns):
        self.annotations += anns

    def compatible_elements(self, e1, e2):
        return (e1.name not in e2.excludes) and (e2.name not in e1.excludes) and (e1.name != e2.name)

    def get_element(self, name):
        for element in self.elements:
            if element.name == name:
                return element

    def get_parents(self, fn):
        return (fn.get_frame(p) for p in self.parents)

    def get_lu(self, lu_name):
        for lu in self.lexicalUnits:
            if lu.name == lu_name:
                return lu

    def add_fe_relation(self, relation):
        self.fe_relations.append(relation)

    def __eq__(self, other):
        return type(self) is type(other) and self.ID, self.name == other.ID, other.name

    def __hash__(self):
        return hash((self.ID, self.name))

    RATTRS = 'name ID relations elements'.split()

    def __repr__(self):
        return 'Frame(%s)' % ', '.join('%s=%s' % (k, getattr(self, k)) for k in Frame.RATTRS)

    def __str__(self):
        return self.name

    # def __str__(self):
    #     # elements = self.format_elements()
    #     formatted = """Name: {name}
    #                 \nFrame Relations: {relations}
    #                 \nElements: {elements}
    #                 \nFrame Element Relations: {fe_relations}
    #                 \nID: {ID}
    #                 \nLUs: {lexicalUnits}""".format(**vars(self))
    #     return formatted

    def format_elements(self):
        return '\n'.join(e.name for e in self.elements)

    def propagate_elements(self):
        """ testing... propagate elements from valences to highest point, e.g. "Theme" instead of "Fluid"
        """
        if len(self.individual_valences) <= 0:
            print("Need to read in the LUs for this frame first.")
            return None
        for valence in self.individual_valences:
            for fe_relation in self.fe_relations:
                if valence.fe == fe_relation.fe2 and fe_relation.name == "Inheritance":
                    valence.fe = fe_relation.fe1
                    # print(valence.fe)
        for group_re in self.group_realizations:
            for valencePattern in group_re.valencePatterns:
                for valence in valencePattern.valenceUnits:
                    for fe_relation in self.fe_relations:
                        if valence.fe == fe_relation.fe2 and fe_relation.name == "Inheritance":
                            valence.fe = fe_relation.fe1


class FrameElement(object):
    def __init__(self, name, abbrev, core, frame_name, ID, semtype=None):
        self.name = name
        self.abbrev = abbrev
        self.coreType = core
        self.frame_name = frame_name
        self.semtype = semtype
        self.excludes = []
        self.requires = []
        self.ID = ID

    def __str__(self):
        return '%s:%s' % (self.frame_name, self.name)

    RATTRS = 'frame_name name ID coreType semtype'.split()

    def __repr__(self):
        return 'FrameElement(%s)' % ', '.join('%s=%s' % (k, getattr(self, k)) for k in FrameElement.RATTRS)

    def __hash__(self):
        return hash((self.ID, self.name))

    def __eq__(self, other):
        return type(self) is type(other) and self.ID, self.name == other.ID, other.name

    def set_semtype(self, semtype):
        self.semtype = semtype

    def add_excludes(self, excluded_element):
        self.excludes.append(excluded_element)

    def add_requires(self, required_element):
        self.requires.append(required_element)


class SemType(object):
    def __init__(self, name, ID):
        self.name = name
        self.ID = ID

    def __str__(self):
        return self.name


def strip_definition(definition):
    # The encoding is necessary for python2.7
    encoded = et.fromstring(definition.encode('utf-8'))
    return ''.join(encoded.itertext())


class FrameBuilder(object):
    def __init__(self, replace_tag):
        self.replace_tag = replace_tag
        self.lu_path = "fndata-1.6/lu/"

    def build_frame(self, xml_path):
        tree = et.parse(xml_path)
        root = tree.getroot()
        name = root.attrib['name']
        ID = int(root.attrib['ID'])
        elements = []
        lexemes = []
        relations = []
        parents = []
        children = []
        definition = ""
        for child in root:
            tag = child.tag.replace(self.replace_tag, "")
            if tag == "FE":
                elements.append(self.build_FE(child, name))
            elif tag == "lexUnit":
                lu = self.build_LU(child, name)
                if lu:
                    lexemes.append(lu)
            elif tag == "frameRelation":
                relation = child.attrib['type']
                related = [r.text for r in child.getchildren()]
                if len(related) > 0:
                    if relation == "Inherits from":
                        parents += related
                    if relation == 'Inherited by':
                        children += related
                    fr = FrameRelation(relation, related)
                    # fr = FrameRelation(atts['type'])
                    # relations.append(fr)
                    relations.append(fr)
            elif tag == "definition":
                xml_def = child.text
                # definition_xml = ET.fromstring(child)
                definition = strip_definition(child.text)
        frame = Frame(name, elements, lexemes, relations, parents, children, definition, xml_def, ID)
        return frame

    def build_FE(self, child, name):
        atts = child.attrib
        element = FrameElement(atts['name'], atts['abbrev'], atts['coreType'], name, int(atts['ID']))
        for c2 in child.getchildren():
            t = c2.tag.replace(self.replace_tag, "")
            if t == "semType":
                s = SemType(c2.attrib['name'], c2.attrib['ID'])
                element.set_semtype(s)
            if t == "excludesFE":
                element.add_excludes(c2.attrib['name'])
            if t == "requiresFE":
                element.add_requires(c2.attrib['name'])
        return element

    def build_LU(self, child, name):
        atts = child.attrib
        s = None
        for c2 in child.getchildren():
            t = c2.tag.replace(self.replace_tag, "")
            if t == "semType":
                s = SemType(c2.attrib['name'], c2.attrib['ID'])
        ID = atts['ID']
        lu = ShallowLU(atts['name'], atts['POS'], name, atts['ID'], atts['status'])
        if s:
            lu.set_semtype(s)
        return lu
        # if atts['status'] != "Problem":
        #     print(name)
        #     path = self.lu_path + "lu{}.xml".format(ID)
        #     lu = self.parse_lu_xml(path)
        #     if s:
        #         lu.set_semtype(s)
        #     return lu


class FrameElementRelation(object):
    """ Defines a relation between two FEs. """

    def __init__(self, fe1, fe2, name, superFrame, subFrame):
        self.fe1        = fe1
        self.fe2        = fe2
        self.name       = name
        self.superFrame = superFrame
        self.subFrame   = subFrame

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(self.__dict__)

    def __str__(self):
        return "{subFrame}:{fe2} <{name}> {superFrame}:{fe1}".format(**vars(self))

    __repr__ = __str__


class FrameRelation(object):
    """Contains relation type (Inchoative Of, etc.) and the associated frames (Cause_motion, etc.).

    If FrameNet.build_relation() has been called, related_frames will point to a list of actual Frame
    objects, vs. strings.
    """

    def __init__(self, relation_type, related_frames):
        self.relation_type = relation_type
        self.related_frames = related_frames

    def __str__(self):
        return self.relation_type

    def __repr__(self):
        related = [frame.name for frame in self.related_frames]
        return "{}: {}".format(self.relation_type, str(related))