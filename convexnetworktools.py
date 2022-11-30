def find_2D_Gate():
    return

def classify(norm_vec_list, biases_list, markers):

    return

def classify_points(norm_vec_list, biases_list, markers):
    """
    determine which points fall into the gate
    :param norm_vec_list: list where each entry np.array of shape (2,)
    :param biases_list: list where each entry np.array of shape (1,)
    :param markers: np.array of shape (2,nr_of_cells) - marker values
    :return:
        one_hot_in_gate: np.array of shape (nr_of_cells,) -
                         one-hot-encoded - 1 means cell in gate
    """
    nr_norm_vec = len(norm_vec_list)
    H_values = value_halfspace(np.array(norm_vec_list), np.array(biases_list), markers)
    sign_matrix = np.sign(H_values)
    one_hot_in_gate = (np.sum(sign_matrix, axis=0) > (nr_norm_vec - 0.5)) * 1
    return one_hot_in_gate

def value_halfspace(normal_vector, bias, cell_value):
    """
    :param normal_vector: torch.tensor of torch.Size([2])
    :param bias: torch.tensor of torch.Size([1])
    :param cell_value: torch.tensor of float of torch.Size([2, M]) where M number of cells
    :return:
       torch.Size([M]) halfspace value of the cells
    """
    return normal_vector @ cell_value + bias