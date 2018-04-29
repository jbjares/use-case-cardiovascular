FROM train:python-deep-learning

COPY metadata.rdf /metadata.rdf
COPY algorithm.py /algorithm.py
COPY query.sparql /query.sparql


