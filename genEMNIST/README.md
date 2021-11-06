This generates the gzip of the pickled EMNIST database. The database is stolen from torch, and is turned into an MNIST-formatted pickled gzip, with the filename `emnist.pkl.gz`.

Because this is a separate utility, it has its own requirements. Install them in a separate venv, *please*.

This will also download a lot of stuff that doesn't end up in the final result.
Feel free to delete: (after you're done generating emnist.pkl.gz)
 - The EMNIST directory and all the files contained therein.
 - The virtualenv you used to construct the whole thing. Torch is kinda big.

To run this utility, it's just `python3 gen_emnist.py`.
