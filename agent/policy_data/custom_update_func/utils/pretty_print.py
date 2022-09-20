from tabulate import tabulate

def pformat_table(data,headers,row_indices,tablefmt="grid",floatfmt="+4.2f"):
    table=tabulate(data, headers=headers,showindex=row_indices,tablefmt=tablefmt,floatfmt=floatfmt)
    return table
    