import re

def extract_quote(sentence):
    """
    Find all the part with the double quote
    : param sentence: the input sentence
    : return : the list of the match part
    """
    # Define a regular expression pattern to match quoted text
    pattern = r'"([^"]*)"'

    # Use re.findall to find all matches of the pattern in the sentence
    matches = re.findall(pattern, sentence)

    return matches