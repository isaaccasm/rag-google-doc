import hashlib


def hash_text(text: str):
    """
    Function to get the hash of a text.
    :param text: The text to hash
    :return:
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
