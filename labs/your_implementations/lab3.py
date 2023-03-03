from typing import List

from lab_utils import LabPredictor

# pylint: disable=pointless-string-statement
"""
Third lab: involving sentiment analysis

Please follow the exercise sheet if you haven't already.
You will implement a sentiment model that you can use here
to alter your preedictions in some way.

It is similar to the previous lab in that sense.
Feel free to use any available functions from NLTK, spaCy and scikit-learn.
"""


class Lab3(LabPredictor):
    def __init__(self):
        super().__init__()

    def predict(self, text) -> List[str]:
        print(f"Lab3 receiving: {text}")
        if not bool(text):
            # use a cold-start strategy from previous labs, or improve upon it
            return ["no", "words"]

        # do all kinds of preprocessing
        
        # use sentiment analysis to alter predictions
        # (and perhaps POS tagging - see exercise sheet)

        return ["this", "is", "lab", "3"]

    def train(self) -> None:
        # train or load your sentiment analysis model
        # as well as the language model of your choice (e.g. trigram)
        pass
