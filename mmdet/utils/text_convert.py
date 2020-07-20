import string


def get_vocabulary(voc_type, EOS='EOS', PADDING='PAD', UNKNOWN='UNK'):
    voc = None
    types = ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    if voc_type == 'LOWERCASE':
        voc = list(string.digits + string.ascii_lowercase)
    elif voc_type == 'ALLCASES':
        voc = list(string.digits + string.ascii_letters)
    elif voc_type == 'ALLCASES_SYMBOLS':
        voc = list(string.printable[:-6])
    else:
        raise KeyError('voc_type must be one of "LOWERCASE", "ALLCASES", "ALLCASES_SYMBOLS"')

    # update the voc with specifical chars
    voc.append(EOS)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char