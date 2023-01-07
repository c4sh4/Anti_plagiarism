""" Compare of couple py files."""
import io
import tokenize
import token
import dis
import argparse
import time
import pandas as pd


def input_txt(input_path):
    """ Open  file with a list of pairs of documents."""
    # 'input.txt' r'C:\Users\Admin\Desktop\Tinkoff_edu\test.txt '
    with open(str(input_path), 'r', encoding='utf-8') as input_file:
        in_path = input_file.read().split()
    return in_path[:20]


def get_file(source):
    """ Get py file from link."""
    with open(str(source), 'r', encoding='utf-8') as py_file:
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
    temp_dict = {'token': tokens_name, 'lexeme': lexeme}
    lexeme_df = pd.DataFrame(temp_dict)
    return lexeme_df


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
    dl_list = []  # Damerau Levenshtein distance
    lcs_list = []  # lcs distance
    while p < len(f_list) - 1:
        dl_d = damerau_levenshtein(f_list[p], f_list[p + 1])
        dl_list.append(round(1 - (dl_d / len(f_list[p])), 3))
        if round(1 - (dl_d / len(f_list[p])), 3) < 0.30:
            lcs = longest_common_subsequence(f_list[p], f_list[p + 1])
            lcs_list.append(round(lcs / len(f_list[p]), 3))
        else:
            lcs_list.append(round(1 - (dl_d / len(f_list[p])), 3))
        p += 2
    temporary_dict = {'dl': dl_list, 'lcs': lcs_list}
    code_sim_df = pd.DataFrame(temporary_dict)
    return code_sim_df


def token_column(lex_list):
    """ Create a column for the final dataframe with token based distance."""
    token_list = []
    for tok in enumerate(lex_list):
        token_list.append(list(lex_list[tok[0]]['token']))
    return code_similarity(token_list)


def lexeme_column(lex_list):
    """ Create a column for the final dataframe with py lexeme based distance."""
    s_lex_list = []
    for lexeme in enumerate(lex_list):
        lex_list[lexeme[0]] = lex_list[lexeme[0]][lex_list[lexeme[0]]['token'] != 'STRING']
    for lex_ in enumerate(lex_list):
        s_lex_list.append(list(lex_list[lex_[0]]['lexeme']))
    return code_similarity(s_lex_list)


def get_scores(f_list, pair_list):
    """ Scoring representations of a pair of files in the form of byte code, tokens and program text.
    Scoring is based on the Damerau Levenshtein distance and the largest common subsequence.
    """
    lexeme_list = []
    bytecode_list = []
    for fc in f_list:
        lexeme_list.append(lexeme_selection(fc))
        bytecode_list.append(bytecode_selection(fc))
    df = pd.DataFrame(pair_list)
    df = df.join(code_similarity(bytecode_list))
    df = df.rename(columns={0: "f", "dl": "b_dl", "lcs": "b_lcs"})
    df = df.join(token_column(lexeme_list))
    df = df.rename(columns={"dl": "t_dl", "lcs": "t_lcs"})
    df['distance'] = abs(df.b_dl - df.t_lcs)
    # Replacement for file pairs for which binary representations were not obtained.
    df.loc[(df['b_dl'] == 0), 'b_dl'] = df.t_dl
    # additional test
    doubt = additional_test(df[(df['distance'] > 0.150)], lexeme_list)
    df['l_dl'] = df['b_dl']
    df['l_lcs'] = df['t_lcs']
    for ind, r in enumerate(doubt['f']):
        df.loc[(df['f'] == r), 'l_dl'] = doubt['l_dl'][ind]
        df.loc[(df['f'] == r), 'l_lcs'] = doubt['l_lcs'][ind]
    df = df.drop(columns='distance')
    res = round((df['b_dl'] + df['t_lcs'] + (df['l_dl'] + df['l_lcs'])/2) / 3, 3)
    for r in enumerate(res):
        res[r[0]] = res[r[0]] if res[r[0]] > 0 else 0.
    result_list = list(res)
    # print(df)
    return result_list


def additional_test(df_doubt, lexeme_list):
    """ Checking pairs of texts on which the byte code similarity check gave a doubtful result or was not passed.
    Files are compared directly for each token, with the exclusion of comments.
    """
    short_lexeme_list = []
    for f in df_doubt.f:
        short_lexeme_list.append(lexeme_list[int(f) - 1])
        short_lexeme_list.append(lexeme_list[int(f)])
    df_doubt = df_doubt.reset_index()
    df_doubt = df_doubt.join(lexeme_column(short_lexeme_list))
    df_doubt = df_doubt.rename(columns={"dl": "l_dl", "lcs": "l_lcs"})
    df_doubt = df_doubt.drop(columns='index')
    return df_doubt


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
    list_of_couples = []
    source = parser()
    path_list = input_txt(source[0])
    # path_list = input_txt()
    for i, path in enumerate(path_list):
        file_list.append(get_file(path))
        if i % 2 == 1:
            list_of_couples.append(i)
    score_list = get_scores(file_list, list_of_couples)
    scores_file(score_list, source[1])
    end = time.time()-start
    print('time of evaluation: ', round(end, 3))
