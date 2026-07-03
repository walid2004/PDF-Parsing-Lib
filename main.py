#pdf parsing library
import zlib
import re

def open_pdf(loc):
    with open (loc, 'rb') as f:
        return f
    