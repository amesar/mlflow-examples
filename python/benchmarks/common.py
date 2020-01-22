
def read_data(data_path, num_records):
    with open(data_path, 'r') as f:
        lines = f.readlines()
    columns = lines[0].split(',')[:-1] # remove label column 'quality'
    columns = [ c.replace('"','') for c in columns]
    print("Columns:",columns)
    lines = lines[1:]
    if num_records is not None:
        lines = lines[:num_records]
    num_records = len(lines)
    print("#Records:",num_records)

    records = []
    for line in lines:
        toks = line.strip().split(',')[:-1]
        r = [float(t) for t in toks ]
        dct = { "columns" : columns, "data" : [r] }
        records.append(dct)

    return records
