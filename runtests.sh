#!/bin/bash

py TestColumn.py

py TestAddDerivedColumns.py
py TestColumnDeriverAbs.py
py TestColumnDeriverDate.py
py TestColumnDeriverLen.py
py TestColumnDeriverMinMaxScaler.py
py TestColumnDeriverRobustScaler.py
py TestColumnDeriverTokenizerCharDecimal.py
py TestColumnDeriverUpper.py
# py  -W ignore TestColumnDeriverSentenceEmbedder.py

py TestMakeNumericColumns.py
py TestColumnNumericerLabelEncoded.py
