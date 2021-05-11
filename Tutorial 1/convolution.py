def convolution(img, kernel, padding=True):
    """ Performs convolution operation given an image and a kernel

        Parameters
        ----------
        img : array_like
        1-channel image
        kernel : array-like
        kernel (filter) for convolution
        
        Returns
        -------
        np.ndarray
        result of the convolution operation
    """
    result = np.zeros_like(img)
    p_size_i = kernel.shape[0] // 2
    p_size_j = kernel.shape[1] // 2

    if padding:
        padded_img = np.zeros((img.shape[0] + 2 * p_size_i, img.shape[1] + 2 * p_size_j))
        i_first = p_size_i
        i_last = padded_img.shape[0] - p_size_i - 1
        j_first = p_size_j
        j_last = padded_img.shape[1] - p_size_j - 1
        padded_img[i_first: i_last + 1, j_first: j_last + 1] = img
    else:
        padded_img = img.copy()
        i_first = p_size_i
        i_last = padded_img.shape[0] - p_size_i - 1
        j_first = p_size_j
        j_last = padded_img.shape[1] - p_size_j - 1
    
    for i in range(i_first, i_last):
        for j in range(j_first, j_last):
            window = padded_img[i - p_size_i: i + p_size_i + 1, j - p_size_j: j + p_size_j + 1]
            res_pix = np.sum(window * kernel)
            result[i - p_size_i, j - p_size_j] = res_pix
    return result