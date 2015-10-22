import mytools

mytools.create_connectivity_mat(sigma_c = 500,
                            N_pre = 8000,
                            N_post = 2000,
                            x_pre = 1,
                            x_post = 4,
                            fixed_in_degree = 0.02,
                            save_to_file = True,
                            filename = "no_name_specified",
                            dir_name = "connectivity_matrices")