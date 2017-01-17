# DEMO: Build list of frames with all their relations
"""
final_relations = [['Frame', 'Inherits from', 'Is Inherited by', 'Perspective on', 'Is Perspectivized in', 'Uses', 'Is Used by', 'Subframe of', 'Has Subframe(s)', 'Precedes', 'Is Preceded by', 'Is Inchoative of', 'Is Causative of', 'See also']]
for frame in fn.frames:
    temp = [frame.name, None, None, None, None, None, None, None, None, None, None, None, None, None]
    for relation in frame.relations:
        if relation.relation_type in final_relations[0]:
            index = final_relations[0].index(relation.relation_type)
            s = [f.name for f in relation.related_frames]
            temp[index] = s
    final_relations.append(temp)

"""

# DEMO, OANA 3: Build 1200x1200 matrix of framesxframes, with relations filling each cell
# TODO: Use numpy?
# import numpy as np

"""
final = []
first_row = [frame.name for frame in fn.frames]
first_row.insert(0, "SPACE")
final.append(first_row)
column = [frame.name for frame in fn.frames]
for value in column:
    print(value)
    new_row = [value]
    for name in first_row:
    #print("Comparing {} with {}".format(name, value))
        frame = fn.get_frame(name)
        if frame:
            s = frame.is_related(value)
            if s:
                new_row.append(s)
            else:
                new_row.append("None")
    final.append(new_row)
"""

"""
for frame in fn.frames:
    print("Building lus and fes for {}".format(frame.name))
    fnb.build_lus_for_frame(frame.name, fn)
    #v1s = [v for v in frame.individual_valences if v.pt in ['DNI', 'INI'] and v.fe in elements_of_interest]
    print(len(frame.individual_valences))
    for v in frame.individual_valences:
        if v.pt in ['DNI', 'INI'] and v.fe in elements_of_interest:
            fe = v.fe
            #valences = [valence for valence in frame.individual_valences if valence.fe == fe and valence.pt not in ['DNI', 'INI']]
            for valence in frame.individual_valences:
                if valence.fe == fe and valence.pt not in ['DNI', 'INI']:
                    lu = valence.lexeme
                    for anno in valence.annotations:
                        sentence, text = anno.sentence, anno.fe_mappings[fe]
                        if "{}.{}".format(text, fe) not in seen:
                            new_line = [sentence, frame.name, lu, text, fe, valence.pt, valence.gf]
                            seen.append("{}.{}".format(text, fe))
                            final.append(new_line)
"""

# DEMO: Gets parents and children
# inheritance = [[f.name, f.children, f.parents] for f in fn.frames]


# DEMO: Build LUS for all frames, put in tuple format with annotations

"""
final = []
for frame in fn.frames:
    print("Building lus for {}.".format(frame.name))
    fnb.build_lus_for_frame(frame.name, fn)
    for i in frame.annotations:
        mini = [i.sentence.encode('utf-8'), i.lu, i.frame]
        for k, v in i.text_to_valence.items():
            mini.append(k.encode('utf-8'))
            mini.append(v.fe.encode('utf-8'))
            mini.append(v.pt.encode('utf-8'))
            if v.pt in ['INI', 'DNI', 'CNI']:
                mini.append("---")
            else:
                mini.append(v.gf)
        final.append(mini)
"""

# DEMO 2: Build LUS for frames, put annotations in tuple format
# New line/row for each valence unit for each annotation
"""
final = []
for frame in fn.frames:
    print("Building lus for {}.".format(frame.name))
    fnb.build_lus_for_frame(frame.name, fn)
    for i in frame.annotations:
        for k, v in i.text_to_valence.items():
            mini = [i.sentence.encode('utf-8'), i.lu.encode('utf-8'), i.frame.encode('utf-8')]
            mini.append(k.encode('utf-8'))
            mini.append(v.fe.encode('utf-8'))
            mini.append(v.pt.encode('utf-8'))
            if v.pt in ['INI', 'DNI', 'CNI']:
                mini.append("---".encode('utf-8'))
            else:
                mini.append(v.gf.encode("utf-8"))
            final.append(mini)
"""

# DEMO: FE Relations

from framenet.builder import build


def main():
    fn, _ = build()

    final = []
    for frame in fn.frames:
        for fe in frame.fe_relations:
            if fe.name in ['Inheritance', 'Perspective_on']:
                final.append([frame.name, fe.superFrame, fe.fe1, fe.name, fe.fe2, fe.subFrame])



                # DEMO: Writing to CSV file
                # resultFile = open("output.csv", "w")
                # wr = csv.writer(resultFile, dialect="excel")
                # wr.writerows(final)


                # DEMO: SCHEMAS
                # schemas = utils.generate_schemas_for_frames(fn.frames)
                # Write these to a file


                # DEMO: CONSTRUCTIONS
                # total = build_cxns_for_frame("Motion", fn, fnb, "Manner", "V")

                # You can then write the values from the total dictionary into files:
                # * cxns_all: all valences converted 1-1 to cxns
                # * cxns_collapsed: valences collapsed into smaller set
                # * tokens: tokens created from these frames ("swarm.v", etc.)
                # * types: type-cxns ("Fluidic_motionType") created from these frames
                # * pp: PP constructions that are specific to the frame (e.g., Instrument-PP)
                # * prep_types: General prepositional type constructions used for frame (E.g., "Instrument-Prep")
                # * prepositions: Preposition constructions that are used for that frame (E.g. "With-Preposition"[subcase of Instrument-Prep])
                # NOTE: The last three (pp, prep_types, and prepositions) are necessary for the collapsed cxns, since this filters by PP-type,
                # and collapses two valence units if they are PPs mapping onto the same FE.


                # DEMO: PREPOSITION CONSTRUCTIONS (distinct from build_cxns_for_frame, above)
                # prepositions = utils.build_prepositions(fn)


if __name__ == '__main__':
    main()