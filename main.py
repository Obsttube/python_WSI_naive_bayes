#!/usr/bin/env python

__copyright__ = "Copyright 2020, Piotr Obst"

import sys
from typing import List, Tuple

from cmc import CMC
from income import Income
from naive_bayes import Feature, NaiveBayes, set_verbosity
from mushroom import Mushroom


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '-v':
        set_verbosity(True)
    CMC.execute()
    Mushroom.execute()  # very well trained - almost always correct results. Probably because of a very predictable dataset
    Income.execute()
