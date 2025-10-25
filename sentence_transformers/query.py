import import_ipynb
import sys
sys.path.append("bert_sbert")  # add the directory to Python path

import bert_sbert  # import the notebook as a module

query_text = "For the Peace of the world"

# now you can access the function
result = bert_sbert.query_verses(query_text)
