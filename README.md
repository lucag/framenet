# framenet
Package for reading in FrameNet data and performing operations on it, such as creating ECG grammars.

## Installation

The packages now supports `pip`. To install (in dev mode), do

```bash
$ pip intstall -e .
```

See more in the [tutorial](https://github.com/icsi-berkeley/framenet/wiki/FrameNet-Querying-Tutorial).

To build a customized FrameNet object, run:

```python
from framenet.builder import build

fn, fnb = build()
```

You can then retrieve a frame object by referencing the "fn" (FrameNet) object:

```python
frame = fn.get_frame("Abandonment")
```

By default, the frames contain shallow information about lexical units (name, POS, etc). To retrieve the valence patterns,
you can use the "FrameNetBuilder" object and pass in the name of the frame, as well as the FrameNet object (fn):

```python
fnb.build_lus_for_frame("Abandonment", fn)
```

Now, the lexicalUnits field for the Abandonment field will contain valence pattern information:

```python
frame.lexicalUnits[0].valences
```
...


