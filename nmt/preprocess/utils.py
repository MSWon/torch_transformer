import re

URL_TOKEN = "<URL>"
URL_REGEX = "(http[s]?://([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)|(www.([a-zA-Z]|[가-힣]|[0-9]|[-_@\.&+!*/])+)"

BUFFER_LINES = 5000000 
CHUNK_SIZE = 10000

LENGTH_RATIO_LOWER_BOUND = 0.5
LENGTH_RATIO_UPPER_BOUND = 2.0
MAX_SENT_LENGTH = 100

def remove_long_sent(sent: str):
    if len(sent) <= MAX_SENT_LENGTH:
        return sent
    return None

def remove_sent_by_length_ratio(src_sent: str, tgt_sent: str):
    length_ratio = len(src_sent) / len(tgt_sent)
    flag = length_ratio > LENGTH_RATIO_LOWER_BOUND and length_ratio < LENGTH_RATIO_UPPER_BOUND
    if flag:
        return src_sent, tgt_sent
    return None, None
