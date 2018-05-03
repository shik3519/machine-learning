from pandas.api.types import is_string_dtype, is_numeric_dtype, is_float_dtype, is_integer_dtype, is_categorical_dtype, \
    is_datetime64_ns_dtype


def type_conversion(df, dt_col=None, conv_to_int=None):
    '''
    input:
    df: dataframe
    dt_col(list of column name): column which needs to be converted to datetime
    conv_to_int(list of column name): float columns which needs to be converted to int

    output:
    df_n: new dataframe, original dataframe is still intact
    '''

    print(df.info(memory_usage='deep'))
    print("=============")

    #     int8 = {'min': -128, 'max': 127}
    #     uint8 = {'min': 0, 'max': 255}
    #     int16 = {'min': -32768, 'max': 32767}
    #     uint16 = {'min': 0, 'max': 65535}
    #     int32 = {'min': -2147483648, 'max': 2147483647}
    #     uint32 = {'min': 0, 'max': 4294967295}
    #     int64 = {'min': -9223372036854775808, 'max': 9223372036854775807}
    #     uint64 = {'min': 0, 'max': 18446744073709551615}

    cols = df.columns
    df_n = df.copy(deep=True)
    cols_non_string = []
    cols_null = []
    if dt_col:
        for c in dt_col:
            if df_n[c].isnull().sum() == 0:
                df_n[c] = pd.to_datetime(df_n[c])
            else:
                cols_null.append(c)
    for c in cols:
        if is_categorical_dtype(df_n[c]):
            continue
        elif is_string_dtype(df_n[c]) and c not in cols_null:
            if df_n[c].isnull().sum() == 0:
                df_n[c] = df_n[c].astype('category')
            else:
                cols_null.append(c)
        else:
            cols_non_string.append(c)
    for c in cols_non_string:
        if df_n[c].isnull().sum():
            cols_null.append(c)
        else:
            if is_integer_dtype(df_n[c]) or c in conv_to_int:
                cmin = df_n[c].min()
                cmax = df_n[c].max()
                if cmin >= 0:
                    if cmax < 256:
                        df_n[c] = df_n[c].astype(np.uint8)
                    elif cmax < 65536:
                        df_n[c] = df_n[c].astype(np.uint16)
                    elif cmax < 4294967296:
                        df_n[c] = df_n[c].astype(np.uint32)
                    else:
                        df_n[c] = df_n[c].astype(np.uint64)
                else:
                    if cmin > -129 and cmax < 128:
                        df_n[c] = df_n[c].astype(np.int8)
                    elif cmin > -32769 and cmax < 32768:
                        df_n[c] = df_n[c].astype(np.int16)
                    elif cmin > -2147483649 and cmax < 2147483648:
                        df_n[c] = df_n[c].astype(np.int32)
                    else:
                        df_n[c] = df_n[c].astype(np.int64)

    print(df_n.info(memory_usage='deep'))
    print("=============")
    print(f'Columns with nulls {set(cols_null)}')
    return df_n

################################################################################

# This function loads a file, resize it and write in the output folder
def img_resize(fname, outdir, sz, in_dir):
    '''
    fname: image filename
    outdir: relative path to output directory
    sz: final size of image
    in_dir: relative path to the input directory
    '''
    os.makedirs(outdir, exist_ok=True)
    im = cv2.imread(in_dir + fname)
    small_im = cv2.resize(im, (sz, sz))
    cv2.imwrite(outdir + fname, small_im)


def parallel_runs_img_resize(data_list, outdir, in_dir, sz=300, process=4):
    '''
    data_list: list of filenames of images stores in a list
    outdir: relative path to output directory
    sz: final size of image
    in_dir: relative path to the input directory
    process: num of threads in your cpu
    '''
    pool = multiprocessing.Pool(processes=process)
    img_resize_x = partial(img_resize, outdir=outdir, sz=sz, in_dir=in_dir)
    pool.map(img_resize_x, data_list)

################################################################################
def bulk_copyfiles(filelist, source, destination, overwrite = True):
    '''
    filelist: list of filenames you need to copy
    source: source directory
    destination: destination directory
    '''
    if os.path.exists(destination) and overwrite: shutil.rmtree(destination)
    os.makedirs(destination, exist_ok=True)
    for fname in filelist:
        if os.path.exists(source + fname):
            shutil.copy(os.path.join(source, fname), destination)

