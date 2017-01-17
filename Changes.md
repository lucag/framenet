# Changes to the original code

I'm documentng the changes that I'm making to Sean's original code. Some
are for the sake of compatibility, others for practicality. 

## Renamed folders/packages

* The root folder `src` became `framenet`.
* Added a package `ecg` containing all the related API, with file names shortedned.
* All package names are singular.
* Added a package `examples` to gather all the miscellaeous "stuff".

## Return type of methods

* Methods like `parents` and `elements` return objects instead of strings.
 
## Compatibility with Python 2
 
* `ecg_utilities.ECGUtilities` is acually a namespace for the methods it
  contains. Python 2 wants static methods be marked, so I removed for compatibility.
  
## Packaging

* Added `setup.py` and the ability to be installed via `pip`.

