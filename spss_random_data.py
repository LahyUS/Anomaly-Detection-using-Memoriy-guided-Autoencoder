import openpyxl
import random
import os
import string
import numpy as np

input_detail = [] #[['SP1', 'SP2', 'SP3' , 'SP4', 'SP5']]
output_excel_path= './user.xlsx'


def output_Excel(input_detail, output_excel_path):
    # Xác định số hàng và cột lớn nhất trong file excel cần tạo
    row = len(input_detail)
    col = len(input_detail[0])

    # Tạo một workbook mới và active nó
    wb = openpyxl.Workbook()
    ws = wb.active

    # Dùng vòng lặp for để ghi nội dung từ input_detail vào file Excel
    for i in range(0, row):
        for j in range(0, col):
            v = input_detail[i][j]
            tab_column = j + 1
            tab_row = i + 1
            ws.cell(tab_row, tab_column, value=v)

    # Lưu lại file Excel
    wb.save(output_excel_path)


def random_input_data(input_mat):
    for i in range(0, 10):
        tmp = []
        a = 1
        b = 5
        for j in range(0, 396):
            # generate random integer between a and b (including a and b)
            randint = random.randint(a, b)
            tmp.append(randint)
        input_mat.append(tmp)

    np_array = np.array(input_mat)
    transpose = np_array.T
    transpose_list = transpose.tolist()

    prop1 = [0, 1, 2, 3, 4]
    prop2 = [0, 1, 2, 3]
    prop3 = [0, 1, 2]
    prop4 = [-1, 0, 1]

    for i in range(0, 396):
        tmp_row = transpose_list[i]
        result = tmp_row

        for j in range(0, len(result)):
            cmp = result[j]
            if j % 2 == 0:
                if cmp == 1:
                    plus = random.choices(prop1, weights=(5, 10, 10, 45, 30), k=5)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 2:
                    plus = random.choices(prop2, weights=(5, 10, 40, 30), k=4)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 3:
                    plus = random.choices(prop3, weights=(10, 20, 20), k=3)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 4:
                    plus = random.choices(prop4, weights=(5, 30, 5), k=3)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 5:
                    plus = random.randint(-1, 0)
                    result[j] = cmp + plus

            elif j % 2 == 1:
                if cmp == 1:
                    plus = random.choices(prop1, weights=(5, 5, 30, 5, 5), k=5)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 2:
                    plus = random.choices(prop2, weights=(5, 35, 10, 5), k=4)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 3:
                    plus = random.choices(prop3, weights=(25, 2, 5), k=3)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 4:
                    plus = random.choices(prop4, weights=(25, 5, 5), k=3)
                    randint = random.choice(plus)
                    result[j] = cmp + randint

                elif cmp == 5:
                    plus = random.randint(-1, 0)
                    result[j] = cmp + plus


        result = random.choices(result, weights=(5, 10, 20, 35, 30, 5, 10, 20, 35, 30), k=10)
        transpose_list[i] = result

    return transpose_list

input_mat = random_input_data(input_detail)
output_Excel(input_mat, output_excel_path)


