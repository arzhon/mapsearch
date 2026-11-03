# mapsearch
Python script to create a CLIP database and search battlemaps using natural language. The script uses 

Before using the map search, a database needs to be created. Add the folder where the maps are located to the variable 
MAPS_FOLDERS = []

and then run the script from the terminal using 
python mapsearch.py index

The script will download the CLIP model from HuggingFace, and then start constructing the library. The duration will be based on the size of the map library. For my own library, it took around 30 min for a library of 10k image files using a MacBook Pro M1 Pro processor. The Python script should be system-agnostic. I initially programmed it to use the M1 processor for the computing, but it should now be usable in Linux and Windows, using Nvidia GPU processing on those systems.

After the index is finished, the search engine can be called by using
python mapsearch.py serve

From the search mini-website one can choose map creators, whether to look for maps, scenes, or tokens (that was something for my own organization. Scenes and Tokens have to be contained within folders containing "Scene" or "Token", anything that is not within a Scene or Token folder will be considered a map).
