""" Compare of couple py files."""
import io
import tokenize
import token
import dis
import argparse
import time


def input_txt(input_path):
    """ Open file with a list of pairs of documents."""
    with open(str(input_path), 'r', encoding='utf-8') as input_file:
        in_path = input_file.read().split()
    return in_path[:10]


def get_file(source_):
    """ Get py file from link."""
    with open(str(source_), 'r', encoding='utf-8') as py_file:
        py_file = py_file.read()
    return py_file


def lexeme_selection(file):
    """ Tokenization of py file."""
    rl_ = io.StringIO(file).readline
    tokens_name = []
    lexeme = []
    for t_type, t_str, (srow, scol), (erow, ecol), pos_str in tokenize.generate_tokens(rl_):
        tokens_name.append(token.tok_name[t_type])
        lexeme.append(t_str)
    tok_dict = {'token': tokens_name, 'lexeme': lexeme}
    return tok_dict


def bytecode_selection(file):
    """ Binary representation of file."""
    instr_list = []
    try:
        file_dis = dis.Bytecode(file)
    except SyntaxError:
        instr_list.append('=)')
        return instr_list
    else:
        for instr in file_dis:
            instr_list.append(instr.opname)
    return instr_list


def longest_common_subsequence(sel1, sel2):
    """ Longest common subsequence."""
    size1 = len(sel1)
    size2 = len(sel2)
    if size1 < size2:
        sel1, sel2 = sel2, sel1
        size1, size2 = size2, size1
    if size1 * size2 == 0:
        return 0
    curr = [0] * (1 + size2)
    for elem1 in sel1:
        prev = list(curr)
        for e in range(size2):
            if elem1 == sel2[e]:
                curr[e + 1] = prev[e] + 1
            else:
                curr[e + 1] = max(curr[e], prev[e + 1])
    return curr[-1]


def damerau_levenshtein(sel1, sel2):
    """ Damerau-levenshtein distance."""
    dict_ = {}
    size1 = len(sel1)
    size2 = len(sel2)
    if size1 * size2 == 0:
        return 0
    for n in range(-1, size1 + 1):
        dict_[(n, -1)] = n + 1
    for m in range(-1, size2 + 1):
        dict_[(-1, m)] = m + 1
    for n in range(size1):
        for m in range(size2):
            if sel1[n] == sel2[m]:
                cost = 0
            else:
                cost = 1
            dict_[(n, m)] = min(
                dict_[(n - 1, m)] + 1,
                dict_[(n, m - 1)] + 1,
                dict_[(n - 1, m - 1)] + cost,
            )
            if n and m and sel1[n] == sel2[m - 1] and sel1[n - 1] == sel2[m]:
                dict_[(n, m)] = min(dict_[(n, m)], dict_[n - 2, m - 2] + 1)
    return dict_[size1 - 1, size2 - 1]


def code_similarity(f_list):
    """ Count the distance between couple files.
    The main metric is the damerau_levenshtein distance.
    If this value is unsatisfactory, lcs is additionally calculated.
    Otherwise, lcs set to the values of based metric.
    """
    p = 0
    dl_ = 0
    lcs_ = 0
    while p < len(f_list) - 1:
        dl_d = damerau_levenshtein(f_list[p], f_list[p + 1])
        dl_ = (round(1 - (dl_d / len(f_list[p])), 3))
        if 0.15 <= round(1 - (dl_d / len(f_list[p])), 3) < 0.60:
            lcs = longest_common_subsequence(f_list[p], f_list[p + 1])
            lcs_ = (round(lcs / len(f_list[p]), 3))
        else:
            lcs_ = (round(1 - (dl_d / len(f_list[p])), 3))
        p += 2
    code_sim_df = [dl_, lcs_]
    return code_sim_df


def token_column(lex_list):
    """ Create a column for the final dataframe with token based distance."""
    token_list = [list(lex_list[0]['token']), list(lex_list[1]['token'])]
    return code_similarity(token_list)


def lexeme_column(lex_list):
    """ Create a column for the final dataframe with py lexeme based distance."""
    s_lex_list1 = []
    s_lex_list2 = []
    pair_lex_list = []
    for lexeme in enumerate(lex_list[0]['token']):
        if lex_list[0]['token'][lexeme[0]] != 'STRING':
            s_lex_list1.append(lex_list[0]['lexeme'][lexeme[0]])
    for lexeme in enumerate(lex_list[1]['token']):
        if lex_list[1]['token'][lexeme[0]] != 'STRING':
            s_lex_list2.append(lex_list[1]['lexeme'][lexeme[0]])
    pair_lex_list.append(list(s_lex_list1))
    pair_lex_list.append(list(s_lex_list2))
    return code_similarity(pair_lex_list)


def get_scores(f_list):
    """ Scoring representations of a pair of files in the form of byte code,
    tokens and program text.
    Scoring is based on the Damerau Levenshtein distance and the largest common subsequence.
    """
    lexeme_list = []
    bytecode_list = []
    sim_list = []

    for fc in f_list:
        lexeme_list.append(lexeme_selection(fc))
        bytecode_list.append(bytecode_selection(fc))

    sim_list.append(code_similarity(bytecode_list))  # b_dl, b_lcs

    sim_list.append(token_column(lexeme_list))  # t_dl, t_lcs

    distance = abs(sim_list[0][0] - sim_list[1][1])

    if sim_list[0][0] == 0:
        sim_list[0][0] = sim_list[1][1]  # b_dl = t_lcs
    if sim_list[0][1] == 0:
        sim_list[0][1] = sim_list[1][0]  # b_lcs = t_dl
    if sim_list[1][0] == sim_list[1][1] == 0:
        sim_list[1][0] = sim_list[0][0]  # t_dl = b_dl
        sim_list[1][1] = sim_list[0][1]  # t_lcs = b_lcs
    if distance > 0.250:
        sim_list.append(lexeme_column(lexeme_list))
    else:
        sim_list.append([sim_list[0][0], sim_list[1][1]])
    res = round(((sim_list[0][0] + sim_list[0][1]) / 2 +
                 (sim_list[1][0] + sim_list[1][1]) / 2 +
                 (sim_list[2][0] + sim_list[2][1]) / 2) / 3, 3)
    res = res if res > 0 else 0.
    return res


def scores_file(res_list, scores_path):
    """ Fill scores.txt."""
    with open(str(scores_path), 'w', encoding='utf-8') as score:
        for res_ in res_list:
            score.write(str(res_) + '\n')


def parser():
    """ Allows you to run compare from a terminal."""
    pars = argparse.ArgumentParser(description='')
    pars.add_argument('input', type=str, help='Input file with a list of file pairs to check.')
    pars.add_argument('scores', type=str, help='Program text similarity evaluation file.')
    args = pars.parse_args()
    source_ = (args.input, args.scores)
    return source_


if __name__ == "__main__":
    start = time.time()
    file_list = []
    score_list = []
    source = parser()
    path_list = input_txt(source[0])
    # path_list = input_txt('input.txt')
    for i, path in enumerate(path_list[:-1]):
        if i % 2 == 0:
            pair_file = [get_file(path), get_file(path_list[i + 1])]
            file_list.append(tuple(pair_file))
    for iter_, pair_ in enumerate(file_list):
        score_list.append(get_scores(pair_))
    scores_file(score_list, source[1])
    # scores_file(score_list, 'score.txt')
    end = time.time() - start
    print('time of evaluation: ', round(end, 3))
