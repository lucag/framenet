"""
This file is intended to be a repository of sample scripts/queries to run over FrameNet. 

@author: Sean Trott

The scripts will be coded as functions, so you can import them into "main" once you run ./build.sh,
as in:
from scripts import retrieve_pt
"""


def retrieve_pt(frame, pt="DNI"):
    """ Requires the lexical units in frame to have already been constructed by FrameNetBuilder,
    so that valence patterns are accessible.
    Returns all valence units with specified phrase type."""
    returned = []
    for lu in frame.lexicalUnits:
        for valence in lu.individual_valences:
            if valence.pt == pt:
                returned.append(valence)
    return returned


def find_cooccurring_fes(frame, elements):
    """Returns a list of FE group realization objects featuring AT LEAST the fes specified in elements.
    ELEMENTS should be a list.
    """
    return [realization for realization in frame.group_realizations if set(elements).issubset(realization.elements)]


def retrieve_fe(frame, fe):
    """Requires the lexical units in frame to have already been constructed by FrameNetBuilder, so that valence patterns are accessible.
    Returns all valence units matching fe.
    """
    return [valence for valence in frame.individual_valences if valence.fe == fe]


def lus_for_frames(frame_set, fn):
    """Very simple function that returns a list of lexical unit objects for each frame in FRAME_SET.
    Input frames in FRAME_SET should be strings, not actual frame objects.

    >> lus_for_frames(['Motion', 'Cause_motion'], fn)
    [[move.v, go.v, ...], [cast.v, catapult.v, ....]]
    """
    return [fn.get_frame(frame).lexicalUnits for frame in frame_set]


def get_valence_patterns(frame):
    patterns = []
    for re in frame.group_realizations:
        patterns += re.valencePatterns
    return patterns


def invert_preps(valences):
    returned = dict()
    for pattern in valences:
        if pattern.pt.split("[")[0].lower() == "pp":
            if pattern.pt not in returned:
                returned[pattern.pt] = []
            if pattern.fe not in returned[pattern.pt]:
                returned[pattern.pt].append(pattern.fe)
    return returned


def find_pattern_frequency(frame, target):
    """ Takes in a frame (with lus already built), and a target "Valence" object. Returns the
    total frequency of that object in frame, which means:
    Total number of annotations across all lus for frame.
    """
    all_valences = all_individual_valences(frame) # Not defined
    # target = all_valences[0]
    return sum(i.total for i in all_valences if i == target)


def pattern_across_frames(frames, target):
    """ Takes in multiple frames (a list) and finds frequency of target across them. """
    returned = dict()
    for frame in frames:
        returned[frame.name] = find_pattern_frequency(frame, target)
    return returned
