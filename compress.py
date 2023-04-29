from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dict = {}
    for i in text:
        if i not in freq_dict:
            freq_dict[i] = 1
        else:
            freq_dict[i] += 1
    return freq_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    # get a list of all the keys
    d_keys = list(freq_dict.keys())
    # if length of frq_dict is 1, create a dummy node and return tree
    if len(freq_dict) == 1:
        if isinstance(d_keys[0], int):
            dummy_node = HuffmanTree((d_keys[0] + 1) % 256)
            return HuffmanTree(None, dummy_node, HuffmanTree(d_keys[0]))
        else:
            return HuffmanTree(None, HuffmanTree(0), HuffmanTree(d_keys[0]))
    # create a queue that has all the nodes of the tree
    # each element in this queue is a list that contains the node itself and
    # its frequency which is how we will keep track of frequencies of combined
    # nodes
    queue = [[HuffmanTree(i), freq_dict[i]] for i in d_keys]
    # while there are still nodes left in the queue
    while len(queue) > 1:
        # get the indexes of the two minimum frequencies
        min1 = _get_min_queue(queue)
        min1 = queue.pop(min1)
        min2 = _get_min_queue(queue)
        min2 = queue.pop(min2)
        # create a huffman tree with the two minimums and add it to the queue
        queue.append([HuffmanTree(None, min1[0], min2[0]), min1[1] + min2[1]])
    # now length of queue should be 1 with only the root node in it
    # return that node
    return queue[0][0]


def _get_min_queue(lst: list[list[HuffmanTree, int]]) -> int:
    """Helper function for build_huffman_tree

    Return index of node with minimum frequency in the <lst> queue
    """
    result = 0
    ctr = 0
    min_freq = lst[0][1]
    while ctr < len(lst):
        if lst[ctr][1] < min_freq:
            min_freq = lst[ctr][1]
            result = ctr
        ctr += 1
    return result


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    if tree.is_leaf():
        return {}
    return _get_codes_helper(tree, {}, '')


def _get_codes_helper(tree: HuffmanTree, d: dict, path: str) -> dict[int, str]:
    """Helper function for get_codes"""
    if tree.symbol is not None:
        # when base case is reached, update the path stored in the dict
        d[tree.symbol] = path
        return path
    else:
        # traverse tree, add 0 when going left and 1 when going right.
        # for each recursive call, pass the updated dictionary
        _get_codes_helper(tree.left, d, path + '0')
        _get_codes_helper(tree.right, d, path + '1')
        return d


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # call helper function
    # this helper function uses a list to keep track of the index instead of an
    # int, so we can use aliasing to keep track of the index through every call
    _number_nodes_helper(tree, [0])


def _number_nodes_helper(tree: HuffmanTree, index: list) -> None:
    """Helper function for number_nodes"""
    if tree is None:
        return
    _number_nodes_helper(tree.left, index)
    _number_nodes_helper(tree.right, index)
    if tree.symbol is None:
        tree.number = index[0]
        index[0] += 1


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    codes = get_codes(tree)
    weighted_sum = 0
    frequency = 0
    for i in codes:
        weighted_sum += len(codes[i]) * freq_dict[i]
        frequency += freq_dict[i]
    return weighted_sum / frequency


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    result = []
    # ctr is set to 7 because we need to traverse each byte backwards
    ctr = 7
    # temporary variable used to store the current byte we are building
    temp = 0
    for i in text:
        code = codes[i]
        for j in code:
            if ctr < 0:
                # once we reach 8 digits, store the current byte and start
                # working on a new one
                result.append(temp)
                temp = 0
                ctr = 7
            new_bit = (int(j) * (2 ** ctr))
            temp += new_bit
            ctr -= 1
    if ctr < 7:
        result.append(temp)
    return bytes(result)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    return _tree_to_bytes_helper(tree, [])


def _tree_to_bytes_helper(tree: HuffmanTree, result: list) -> bytes:
    if tree.is_leaf():
        # for base case, if we reach a leaf, we don't need to do anything
        return
    else:
        # traverse tree in postorder
        _tree_to_bytes_helper(tree.left, result)
        _tree_to_bytes_helper(tree.right, result)
        # for each internal node
        if tree.symbol is None:
            # update the result with current nodes details
            if tree.left.symbol is not None and tree.right.symbol is not None:
                result.extend([0, tree.left.symbol, 0, tree.right.symbol])
            elif tree.left.symbol is not None:
                result.extend([0, tree.left.symbol, 1, tree.right.number])
            elif tree.right.symbol is not None:
                result.extend([1, tree.left.number, 0, tree.right.symbol])
            else:
                result.extend([1, tree.left.number, 1, tree.right.number])
        return bytes(result)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    # if cases are all possibilities of how the node may look like
    # if both subtrees are leaves, return tree with those subtrees
    if node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 0:
        return HuffmanTree(None, HuffmanTree(node_lst[root_index].l_data),
                           HuffmanTree(node_lst[root_index].r_data))
    # if left is a left right is not, return tree with left leaf value and
    # recursive call on right tree
    elif node_lst[root_index].l_type == 0 and node_lst[root_index].r_type == 1:
        rtree = generate_tree_general(node_lst, node_lst[root_index].r_data)
        return HuffmanTree(None,
                           HuffmanTree(node_lst[root_index].l_data), rtree)
    # if left is not a leaft and right is, return tree with right leaf value
    # and recursive call on left tree
    elif node_lst[root_index].l_type == 1 and node_lst[root_index].r_type == 0:
        ltree = generate_tree_general(node_lst, node_lst[root_index].l_data)
        return HuffmanTree(None, ltree,
                           HuffmanTree(node_lst[root_index].r_data))
    # otherwise if both subtrees are not leafs, then return tree with recursive
    # calls on each
    else:
        ltree = generate_tree_general(node_lst, node_lst[root_index].l_data)
        rtree = generate_tree_general(node_lst, node_lst[root_index].r_data)
        return HuffmanTree(None, ltree, rtree)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    code = ''
    result = []
    for i in text:
        code += byte_to_bits(i)
    byte = _make_codes(tree, code)
    for i in range(size):
        result.append(byte[i])
    return bytes(result)


def _make_codes(tree: HuffmanTree, code: str) -> list:
    """Helper function for decompress_bytes"""
    result = []
    temp = tree
    ctr = 0
    while ctr < len(code):
        if code[ctr] == '0':
            temp = temp.left
        elif code[ctr] == '1':
            temp = temp.right
        if temp.is_leaf():
            result.append(temp.symbol)
            temp = tree
        ctr += 1
    return result


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        print(node_lst)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
