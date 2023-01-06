import numpy as np
import pandas as pd
import ast
import io
import pprint
import tokenize
import token
import dis


#
def input_txt():
    # 'input.txt' r'C:\Users\Admin\Desktop\Tinkoff_edu\test.txt'
    with open(r'C:\Users\Admin\Desktop\Tinkoff_edu\test.txt', encoding='utf-8') as f:
        in_path = f.read().split()
        # print(len(in_path[:50]))
    return in_path[:25]


#
def get_file(source):
    py_file = open(str(source), encoding='utf-8').read()
    return py_file


#
def lexeme_selection(file):
    rl = io.StringIO(file).readline
    tokens_name = []
    lexeme = []
    for t_type, t_str, (br, bc), (er, ec), logl in tokenize.generate_tokens(rl):
        tokens_name.append(token.tok_name[t_type])
        lexeme.append(t_str)
    d = {'token': tokens_name, 'lexeme': lexeme}
    df = pd.DataFrame(d)
    return df


#
def bytecode_selection(file):
    instr_list = []
    try:
        file_dis = dis.Bytecode(file)
    except:
        instr_list.append('=)')
        return instr_list
    else:
        for instr in file_dis:
            instr_list.append(instr.opname)
        return instr_list


# longest common subsequence
def longest_common_subsequence(sel1, sel2):
    size1 = len(sel1)
    size2 = len(sel2)
    '''
    if size1 < size2:
        sel1, sel2 = sel2, sel1
        size1, size2 = size2, size1
    '''
    if size1*size2 == 0:
        return 0
    curr = [0]*(1 + size2)
    for elem1 in sel1:
        prev = list(curr)
        for e in range(size2):
            if elem1 == sel2[e]:
                curr[e + 1] = prev[e] + 1
            else:
                curr[e + 1] = max(curr[e], prev[e + 1])
    return curr[-1]


def damerau_levenshtein(sel1, sel2):
    d = {}
    size1 = len(sel1)
    size2 = len(sel2)
    if size1*size2 == 0:
        return 0
    for i in range(-1, size1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, size2 + 1):
        d[(-1, j)] = j + 1
    for i in range(size1):
        for j in range(size2):
            if sel1[i] == sel2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,
                d[(i, j - 1)] + 1,
                d[(i - 1, j - 1)] + cost,
            )
            if i and j and sel1[i] == sel2[j - 1] and sel1[i - 1] == sel2[j]:
                d[(i, j)] = min(d[(i, j)], d[i - 2, j - 2] + 1)  # transposition
    return d[size1 - 1, size2 - 1]


def code_similarity(f_list):
    p = 0
    dl_list = []  # Damerau Levenshtein distance
    lcs_list = []  # lcs distance
    while p < len(f_list) - 1:
        dl = damerau_levenshtein(f_list[p], f_list[p + 1])
        dl_list.append(round(1 - (dl / len(f_list[p])), 3))
        if round(1 - (dl / len(f_list[p])), 3) < 0.30:
            lcs = longest_common_subsequence(f_list[p], f_list[p + 1])
            lcs_list.append(round(lcs / len(f_list[p]), 3))
        else:
            lcs_list.append(round(1 - (dl / len(f_list[p])), 3))
        # print('step: ', p)
        p += 2
    # print('dl_list size is: ', len(dl_list))
    # print('lcs size is: ', len(lcs_list))
    dt = {'dl': dl_list, 'lcs': lcs_list}
    code_sim_df = pd.DataFrame(dt)
    return code_sim_df


def token_column(lex_list):
    token_list = []
    for to in range(len(lex_list)):
        token_list.append(list(lex_list[to]['token']))
    # print('len of token_list is: ', len(token_list))
    return code_similarity(token_list)


def lexeme_column(lex_list):
    s_lex_list = []
    for lexeme in range(len(lex_list)):
        lex_list[lexeme] = lex_list[lexeme][lex_list[lexeme]['token'] != 'STRING']
    for le in range(len(lex_list)):
        s_lex_list.append(list(lex_list[le]['lexeme']))
    # print('len of lex_list is: ', len(s_lex_list))
    return code_similarity(s_lex_list)


if __name__ == "__main__":
    file_list = []
    lexeme_list = []
    bytecode_list = []
    pair_list = []
    path_list = input_txt()
    for i, path in enumerate(path_list):
        file_list.append(get_file(path))
        if i % 2 == 1:
            pair_list.append(i)
    # print('file list size:', len(file_list))

    for fc in file_list:
        lexeme_list.append(lexeme_selection(fc))
        bytecode_list.append(bytecode_selection(fc))
    df = pd.DataFrame(pair_list)

    # print('bytecode: ')
    df = df.join(code_similarity(bytecode_list))
    df = df.rename(columns={0: "f", "dl": "b_dl", "lcs": "b_lcs"})
    # print('tokens: ')

    df = df.join(token_column(lexeme_list))
    df = df.rename(columns={"dl": "t_dl", "lcs": "t_lcs"})
    df['distance'] = abs(df.b_dl-df.t_lcs)

    df.loc[(df['b_dl'] == 0), 'b_dl'] = df.t_dl

    df_doubt = df[(df['distance'] > 0.150)]
    short_lexeme_list = []
    for f in df_doubt.f:
        short_lexeme_list.append(lexeme_list[int(f)-1])
        short_lexeme_list.append(lexeme_list[int(f)])

    df_doubt = df_doubt.reset_index()
    df_doubt = df_doubt.join(lexeme_column(short_lexeme_list))
    df_doubt = df_doubt.rename(columns={"dl": "l_dl", "lcs": "l_lcs"})

    # df = df.join(lexeme_column(lexeme_list))
    # df = df.rename(columns={"dl": "l_dl", "lcs": "l_lcs"})
    df_doubt = df_doubt.drop(columns='index')
    df['l_dl'] = df['b_dl']
    df['l_lcs'] = df['t_lcs']

    for ind, r in enumerate(df_doubt['f']):
        df.loc[(df['f'] == r), 'l_dl'] = df_doubt['l_dl'][ind]
        df.loc[(df['f'] == r), 'l_lcs'] = df_doubt['l_lcs'][ind]
        # df[(df['f'] == r)]['l_dl'] = df_doubt[(df_doubt['f'] == r)]['l_dl']
        # print(df_doubt['l_dl'][ind])

    df = df.drop(columns='distance')
    print(df)
    result_list = []
    df.insert(0, 'result', (df['b_dl'] + df['t_lcs'] + df['l_dl'] + df['l_lcs'])/4)
    print(df['result'])
    print(result_list)
