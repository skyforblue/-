def detectoutliers(list):
    print(type(list))
    outlier_indices = []
    # iterate over features(columns)
 
        # 1st quartile (25%)
    Q1 = np.percentile(list, 25)
    # 3rd quartile (75%)
    Q3 = np.percentile(list,75)
    # Interquartile range (IQR)
    IQR = Q3 - Q1
    # outlier step
    outlier_step = 1.5 * IQR
    # Determine a list of indices of outliers for feature col
    outlier_list_col = list[(list < Q1 - outlier_step) | (list > Q3 + outlier_step )]
 
    return outlier_list_col