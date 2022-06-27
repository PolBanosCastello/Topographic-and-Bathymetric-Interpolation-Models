import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from rich import print as rprint
import numpy as np
import os
from tqdm.auto import tqdm
import math
from concurrent import futures
from rich.progress import Progress
import multiprocessing as mp


def cm_to_inch(value):
    return value/2.54


plt.rcParams["figure.figsize"] = [cm_to_inch(40), cm_to_inch(20)]


class RBI:
    point_folder = f'./Points/Subset_Vilanova/'
    if not os.path.exists(point_folder):
        os.mkdir(point_folder)
    else:
        pass

    max_worker = mp.cpu_count() * 2

    @staticmethod
    def _multiprocess_index_handler(index, handler, args):
        """
        This function adds the index to the return of the handler function. Useful to sort the results of a
        multi-threaded operation
        :param index: index to be returned
        :param handler: function handler to be called
        :param args: list with arguments of the function handler
        :return: tuple with (index, xxx) where xxx is whatever the handler function returned
        """
        result = handler(*args)  # call the handler
        return index, result  # add index to the result

    def multiprocess(self, arg_list, handler, max_workers=20, text: str = "progress..."):
        """
        Splits a repetitive task into several processes
        :param arg_list: each element in the list will crate a thread and its contents passed to the handler
        :param handler: function to be invoked by every thread
        :param max_workers: Max processes to be launched at once
        :return: a list with the results (ordered as arg_list)
        :param text: text to be displayed in the progress bar
        """
        index = 0  # thread index
        ctx = mp.get_context('spawn')
        with futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            processes = []  # empty thread list
            results = []  # empty list of thread results
            for args in arg_list:
                # submit tasks to the executor and append the tasks to the thread list
                processes.append(executor.submit(self._multiprocess_index_handler, index, handler, args))
                index += 1

            with Progress() as progress:  # Use Progress() to show a nice progress bar
                task = progress.add_task(text, total=index)
                for future in futures.as_completed(processes):
                    future_result = future.result()  # result of the handler
                    results.append(future_result)
                    progress.update(task, advance=1)

            # sort the results by the index added by __threadify_index_handler
            sorted_results = sorted(results, key=lambda a: a[0])

            final_results = []  # create a new array without indexes
            for result in sorted_results:
                final_results.append(result[1])
            return final_results

    def __init__(self, data_frame, df_validation, resolution_factor=10, reduction_scale=1):
        self.df = pd.DataFrame(data_frame.values, columns=['X', 'Y', 'Z'])
        self.reduction_scale = reduction_scale

        self.df['X'] = self.df['X'] * self.reduction_scale
        self.df['Y'] = self.df['Y'] * self.reduction_scale

        self.num_points = len(self.df)
        self.resolution_factor = resolution_factor
        self.df_validation = df_validation
        self.df_validation['X'] = self.df_validation['X'] * self.reduction_scale
        self.df_validation['Y'] = self.df_validation['Y'] * self.reduction_scale

        x_min, x_max = int(np.round(min(self.df['X']))), int(np.round(max(self.df['X'])))
        y_min, y_max = int(np.round(min(self.df['Y']))), int(np.round(max(self.df['Y'])))

        num_resolution = ((x_max - x_min) * self.resolution_factor)
        x = np.linspace(x_min, x_max, num=int(num_resolution))
        y = np.linspace(y_min, y_max, num=int(num_resolution))

        for a in range(0, len(self.df_validation['X'])):
            x = np.append(x, self.df_validation['X'][a])
            y = np.append(y, self.df_validation['Y'][a])

        self.df['X'] = self.df['X'] / self.reduction_scale
        self.df['Y'] = self.df['Y'] / self.reduction_scale

        self.df_validation['X'] = self.df_validation['X'] / self.reduction_scale
        self.df_validation['Y'] = self.df_validation['Y'] / self.reduction_scale

        x = x / self.reduction_scale
        y = y / self.reduction_scale

        rprint(f'Min-Max X:  {round(min(x), 2)}  -  {round(max(x), 4)}')
        rprint(f'Min-Max Y:  {round(min(y), 2)}  -  {round(max(y), 4)}')
        rprint(f'Size X: {x.size}   -   Size Y: {y.size}')

        self.xx, self.yy = np.meshgrid(x, y)
        self.zz = np.empty(self.xx.shape)
        self.zz[:] = np.nan

        plt.scatter(self.xx, self.yy)
        plt.show()

    def k_ij(self, x_i, y_i, x_j, y_j, equation, beta):

        if equation == 'exp':
            return math.exp(-(beta * np.linalg.norm(math.dist((x_i, y_i), (x_j, y_j)))) ** 2)
        if equation == 'multi_sqrt':
            if math.dist((x_i, y_i), (x_j, y_j)) != np.linalg.norm(math.dist((x_i, y_i), (x_j, y_j))):
                print(math.dist((x_i, y_i), (x_j, y_j)), '---', np.linalg.norm(math.dist((x_i, y_i), (x_j, y_j))))
            return np.sqrt(1 + ((beta * math.dist((x_i, y_i), (x_j, y_j))) ** 2))

    def calculate_k_matrix(self, points, beta, kernel_function):
        k_matrix = []
        z_vector = []

        for i in range(0, len(points['X'])):
            k_vector = []
            z_vector.append([points['Z'][i]])
            t_sum = 0
            for j in range(0, len(points['X'])):

                k_vector.append(self.k_ij(points['X'][i], points['Y'][i], points['X'][j], points['Y'][j],
                                          beta=beta, equation=kernel_function))

            k_matrix.append(k_vector)

        return np.matrix(k_matrix), np.matrix(z_vector), np.linalg.solve(np.matrix(k_matrix), np.matrix(z_vector))

    def execute_method_no_multiprocessing(self, beta, kernel_function, file_name):
        rprint(f'Execute Radial Basis Function Interpolation with Beta  = {beta}')

        M_k, V_z, V_w = self.calculate_k_matrix(self.df, beta=beta, kernel_function=kernel_function)
        rprint(f'Process of RBF method with Beta = {beta} ...')
        for i in tqdm(range(0, self.zz.shape[0])):
            # t = time.time()
            # rprint(f'{i:04} - {self.zz.shape[0]:04}')
            for j in range(0, self.zz.shape[1]):
                sum_wk = [V_w[a] * self.k_ij(self.xx[i, j], self.yy[i, j], self.df['X'][a], self.df['Y'][a],
                                             beta=beta, equation=kernel_function)
                          for a in range(0, len(self.df['X']))]
                self.zz[i, j] = sum(sum_wk)
            # rprint(time.time() - t)

        rprint('Interpolation Finish.\nTransform Mesh into DataFrame')

        list_p = []
        for a in range(0, self.zz.shape[0]):
            for b in range(0, self.zz.shape[1]):
                P = [self.xx[a, b], self.yy[a, b], self.zz[a, b]]
                list_p.append(P)

        self.df_new = pd.DataFrame(list_p, columns=['X', 'Y', 'Z'])
        self.save_method(file_name, beta=beta)

        rprint(f'Finish Beta  = {beta}')

        return self.df_new

    def execute_method(self, beta, kernel_function, file_name, show_prints=False):
        rprint(f'Execute Radial Basis Function Interpolation with Beta  = {beta}')

        self.M_k, self.V_z, self.V_w = self.calculate_k_matrix(self.df, beta=beta, kernel_function=kernel_function)

        argument_list = []
        list = np.arange(0, self.max_worker + 1)
        index = 1
        for i in list:
            if i != list[-1]:
                print(i)
                inici = int(self.zz.shape[0] / self.max_worker) * (index - 1)
                final = int(self.zz.shape[0] / self.max_worker) * index
                print(inici, final, '\n')
                index += 1
            else:
                print(i)
                inici = int(self.zz.shape[0] / self.max_worker) * self.max_worker
                final = self.zz.shape[0]
                print(inici, final, '\n')
            myargs = [beta, kernel_function, file_name, inici, final]
            argument_list.append(myargs)

        if show_prints:
            rprint(argument_list)

        rprint(f'Multiprocessing of RBF method with Beta = {beta} ...')

        results = self.multiprocess(argument_list, self.execute_method_multiprocessing, max_workers=self.max_worker)

        if show_prints:
            rprint(results)

        list_res = []
        for result in results:
            list_res += result

        if show_prints:
            rprint(list_res, len(list_res))

        zz = np.array(list_res).reshape(self.zz.shape[1], self.zz.shape[0])

        rprint('Interpolation Finish.\nTransform Mesh into DataFrame')

        list_p = []
        for a in range(0, self.zz.shape[0]):
            for b in range(0, self.zz.shape[1]):
                P = [self.xx[a, b], self.yy[a, b], zz[a, b]]
                list_p.append(P)

        self.df_new = pd.DataFrame(list_p, columns=['X', 'Y', 'Z'])
        self.save_method(file_name, beta=beta)

        rprint(f'Finish Beta  = {beta}')

        return self.df_new

    def execute_method_multiprocessing(self, beta, kernel_function, file_name, start, end, show_prints=False):

        rprint(f'[green]Start Multiprocessing Interpolation with Beta {beta} --> {start} - {end} -- {self.zz.shape[0]}')

        list_results = []
        for i in tqdm(range(start, end)):
            for j in range(0, self.zz.shape[1]):
                if show_prints:
                    print(i, j)
                sum_wk = [self.V_w[a] * self.k_ij(self.xx[i, j], self.yy[i, j], self.df['X'][a], self.df['Y'][a],
                                             beta=beta, equation=kernel_function)
                          for a in range(0, len(self.df['X']))]
                list_results.append(sum(sum_wk))

        rprint(f'[yellow]Finish Multiprocessing Interpolation with Beta {beta} --> {start} - {end} -- {self.zz.shape[0]}')

        return list_results

    def save_method(self, name, beta):

        self.df_new.to_csv(f'{self.point_folder}RBI_optim_method_{name}_Beta_{beta}_{self.num_points}_points.csv',
                           index=False, header=False)

    def load_last_method(self, name, beta):

        self.df_new = pd.read_csv(
            f'{self.point_folder}RBI_optim_method_{name}_Beta_{beta}_{self.num_points}_points.csv',
            header=None)
        self.df_new.columns = ['X', 'Y', 'Z']

        return self.df_new

    def calculate_error(self, name_save, method='MSE', show_plot=False):

        def find_nearest(df_predict, value_x, value_y, value_z, min_z_validate):
            if value_z == 0:
                value_z = 0.00001

            df_predict_x_array = df_predict.X.to_numpy()
            df_predict_y_array = df_predict.Y.to_numpy()

            idx_x = (min(np.abs(df_predict_x_array - value_x)))
            idx_y = (min(np.abs(df_predict_y_array - value_y)))

            for x_val in df_predict_x_array:
                if np.abs(x_val - value_x) == idx_x:
                    x = x_val

            for y_val in df_predict_y_array:
                if np.abs(y_val - value_y) == idx_y:
                    y = y_val

            df_aprox = df_predict[(df_predict['X'] == x) & (df_predict['Y'] == y)]
            array_aprox = df_aprox.to_numpy()

            escal_z = abs(min_z_validate) * 1.2
            val_z_escal = (value_z + escal_z)
            int_z_escal = (array_aprox[0][2] + escal_z)
            e_a_z_escal = abs(val_z_escal - int_z_escal)
            e_r_z_escal = (e_a_z_escal / abs(val_z_escal)) * 100

            lst = [value_x, value_y, value_z,
                   array_aprox[0][0], array_aprox[0][1], array_aprox[0][2],
                   abs(value_x - array_aprox[0][0]), abs(value_y - array_aprox[0][1]), e_a_z_escal,
                   ((abs(value_x - array_aprox[0][0])/abs(value_x))*100),
                   ((abs(value_y - array_aprox[0][1])/abs(value_y))*100),
                   e_r_z_escal,
                   [value_z, array_aprox[0][2], (abs(value_z - array_aprox[0][2])),
                    ((abs(value_z - array_aprox[0][2])/abs(value_z))*100)]]
            return lst

        comparative = []
        for i in tqdm(range(0, len(self.df_validation.values))):
            comparative.append(find_nearest(self.df_new, self.df_validation.X[i], self.df_validation.Y[i],
                                            self.df_validation.Z[i], min_z_validate=min(self.df_validation.Z)))

        df_comparative = pd.DataFrame(comparative,
                                      columns=['X_Real', 'Y_Real', 'Z_Real',
                                               'X_Interp', 'Y_Interp', 'Z_Interp',
                                               'X_Error_Absolute', 'Y_Error_Absolute', 'Z_Error_Absolute',
                                               'X_Error_Relative', 'Y_Error_Relative', 'Z_Error_Relative',
                                               'List_Real_Zs'])
        df_comparative.to_csv(f'{self.point_folder}df_comparative_{name_save}_{self.num_points}_points.csv',
                              index=False)

        sct = plt.scatter(df_comparative['X_Real'], df_comparative['Y_Real'], s=12,
                          c=df_comparative['Z_Error_Absolute'], cmap='plasma')
        plt.axis('scaled')
        plt.colorbar(sct)
        plt.title(f'{name_save} Absolute Error')
        plt.savefig(f'{self.point_folder}Plot_Error_Absolute_{method}_{name_save}.png')
        if show_plot:
            plt.show()
        plt.close()

        sct = plt.scatter(df_comparative['X_Real'], df_comparative['Y_Real'], s=12,
                          c=df_comparative['Z_Error_Relative'], cmap='plasma')
        plt.axis('scaled')
        col_bar = plt.colorbar(sct)
        col_bar.ax.get_yaxis().labelpad = 15
        col_bar.ax.set_ylabel('  %  ', rotation=270)
        plt.title(f'{name_save} Relative Error')
        plt.savefig(f'{self.point_folder}Plot_Error_Relative_{method}_{name_save}.png')
        if show_plot:
            plt.show()
        plt.close()

        if method == 'MSE':
            ''' 1/n * sum((y - y')**2) '''
            e_abs_x = 0
            e_abs_y = 0
            e_abs_z = 0

            e_rel_x = 0
            e_rel_y = 0
            e_rel_z = 0

            for i in tqdm(range(0, len(df_comparative))):
                # print(df_comparative.X_Dist[i], df_comparative.Y_Dist[i], df_comparative.Z_Dist[i])
                e_abs_x += df_comparative.X_Error_Absolute[i] ** 2
                e_abs_y += df_comparative.Y_Error_Absolute[i] ** 2
                e_abs_z += df_comparative.Z_Error_Absolute[i] ** 2

                e_rel_x += df_comparative.X_Error_Relative[i] ** 2
                e_rel_y += df_comparative.Y_Error_Relative[i] ** 2
                e_rel_z += df_comparative.Z_Error_Relative[i] ** 2

            return ((e_abs_x / len(df_comparative)), (e_abs_y / len(df_comparative)), (e_abs_z / len(df_comparative))),\
                   ((e_rel_x / len(df_comparative)), (e_rel_y / len(df_comparative)),
                    (e_rel_z / len(df_comparative))), df_comparative

        if method == 'RMSE':
            ''' sqrt(1/n * sum((y - y')**2)) '''
            e_abs_x = 0
            e_abs_y = 0
            e_abs_z = 0

            e_rel_x = 0
            e_rel_y = 0
            e_rel_z = 0

            for i in tqdm(range(0, len(df_comparative))):
                # print(df_comparative.X_Dist[i], df_comparative.Y_Dist[i], df_comparative.Z_Dist[i])
                e_abs_x += df_comparative.X_Error_Absolute[i] ** 2
                e_abs_y += df_comparative.Y_Error_Absolute[i] ** 2
                e_abs_z += df_comparative.Z_Error_Absolute[i] ** 2

                e_rel_x += df_comparative.X_Error_Relative[i] ** 2
                e_rel_y += df_comparative.Y_Error_Relative[i] ** 2
                e_rel_z += df_comparative.Z_Error_Relative[i] ** 2

            return ((math.sqrt(e_abs_x / len(df_comparative))), (math.sqrt(e_abs_y / len(df_comparative))),
                    (math.sqrt(e_abs_z / len(df_comparative)))), \
                   ((math.sqrt(e_rel_x / len(df_comparative))), (math.sqrt(e_rel_y / len(df_comparative))),
                    (math.sqrt(e_rel_z / len(df_comparative)))), df_comparative


if __name__ == "__main__":

    point_folder = f'./Points/Subset_Vilanova/'
    if not os.path.exists(point_folder):
        os.mkdir(point_folder)
    else:
        pass

    point_folder_dataset = f'./Points/'
    if not os.path.exists(point_folder_dataset):
        os.mkdir(point_folder_dataset)
    else:
        pass

    FILE_NAME = 'vilanova'
    RES_FACT = 0.15
    SCALE_FACT = 0.15
    list_beta = [0.10, 0.25, 0.50, 0.75, 1.00, 2.50, 5.00] #, 5.00, 10.0, 25.0, 50.0, 75.0]
    # list_beta = [75.0]
    ERROR_METHOD = 'RMSE'
    KERNEL_FUNC = 'multi_sqrt'
    DO_INTERPOLATE = True
    CALCULATE_ERROR = True
    SHOW_PLOTS = False

    df = pd.read_csv(f'{point_folder_dataset}{FILE_NAME}.csv', sep=';')
    rprint(df)

    plt.scatter(df['X'], df['Y'], c=df['Z'], cmap='plasma')
    plt.title('Input Data Cap de Creus')
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    col_bar = plt.colorbar()
    col_bar.ax.set_ylabel('  Meters  ')
    plt.axis('scaled')
    plt.savefig(f'{point_folder}Plot_Input_Data.png')
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    num_of_points = int(len(df) / 1.10)
    df_subset = df.sample(n=num_of_points, random_state=1)
    df_drop = pd.DataFrame(df.drop(df_subset.index).values, columns=['X', 'Y', 'Z'])

    fit_RBI = RBI(df_subset, df_drop, resolution_factor=RES_FACT, reduction_scale=SCALE_FACT)

    # ddff = fit_RBI.execute_method(beta=BETA, file_name=FILE_NAME)

    if DO_INTERPOLATE:
        dfs = []
        rprint(' ----------------------   Execute Method   ----------------------')

        for BETA in list_beta:
            rprint(f'Number of Subset points are {len(df_subset)}, '
                   f'and the number of Validation points are {len(df_drop)}')
            rprint()
            rprint('  -----   Variables   ----- ')
            rprint('File Name:              ', FILE_NAME)
            rprint('Resolution Factor:      ', RES_FACT)
            rprint('Factor Scale:           ', SCALE_FACT)
            rprint('Constant Beta:          ', BETA)

            df = fit_RBI.execute_method(beta=BETA, kernel_function=KERNEL_FUNC, file_name=FILE_NAME)
            dfs.append(df)

        rprint(dfs)

    if CALCULATE_ERROR:
        rprint(' ----------------------  Calculate Errors  ----------------------')
        dic_errors = {
            "RBF Beta": {}
        }

        for BETA in list_beta:
            rprint('Calculating Errors of Beta', BETA)

            ddff = fit_RBI.load_last_method(FILE_NAME, beta=BETA)

            plt.scatter(ddff['X'], ddff['Y'], c=ddff['Z'], cmap='plasma')
            plt.title(f'Output Data Vilanova Beta {BETA}')
            plt.ylabel('Latitude')
            plt.xlabel('Longitude')
            col_bar = plt.colorbar()
            col_bar.ax.set_ylabel('  Meters  ')
            plt.axis('scaled')
            plt.savefig(f'{point_folder}Plot_Output_Data_Vilanova_Beta_{BETA}.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

            abs_error, rel_error, df_error = fit_RBI.calculate_error(method=ERROR_METHOD, name_save=f'RBF_Beta_{BETA}')
            print(f'Error Absolut XYZ {ERROR_METHOD} RBF Beta = {BETA}:       ', abs_error[0], abs_error[1],
                  abs_error[2])
            print(f'Error Relatiu XYZ {ERROR_METHOD} RBF Beta = {BETA}:       ', rel_error[0], rel_error[1],
                  rel_error[2])

            vec_norm_real = np.linalg.norm(df_error['Z_Real'])
            vec_norm_interpolate = np.linalg.norm(df_error['Z_Interp'])

            print('La Norma del Vector Z_Real:                ', round(vec_norm_real, 4))
            print('La Norma del Vector Z_Interp:              ', round(vec_norm_interpolate, 4))
            print('Diferencia de Normas Real - Interp:        ', round((vec_norm_real - vec_norm_interpolate), 4))
            print('(Norma Real - Norma Interp)/(Norma Real):  ',
                  round(((vec_norm_real - vec_norm_interpolate)/vec_norm_real), 4))

            dic = {
                "Absolute Error": {
                    "X": np.round(abs_error[0], 7),
                    "Y": np.round(abs_error[1], 7),
                    "Z": np.round(abs_error[2], 7),
                },
                "Relative Error": {
                    "X": np.round(rel_error[0], 7),
                    "Y": np.round(rel_error[1], 7),
                    "Z": np.round(rel_error[2], 7)
                },
                "Norma Real": np.round(vec_norm_real, 7),
                "Norma Interp": np.round(vec_norm_interpolate, 7),
                "(Norma Real) - (Interp)": np.round((vec_norm_real - vec_norm_interpolate), 7),
                "(Norma Real - Norma Interp)/(Norma Real)": np.round(
                    ((vec_norm_real - vec_norm_interpolate) / vec_norm_real), 7),
                "Norma(Vector Reals - Vector Interp)": np.round(
                    (np.linalg.norm((df_error['Z_Real'] - df_error['Z_Interp']))), 7)
            }

            dic_to_print = {
                "Absolute Error": np.round(abs_error[2], 7),
                "Relative Error": np.round(rel_error[2], 7),
                "Norma(Vector Reals - Vector Interp)": np.round(
                    (np.linalg.norm((df_error['Z_Real'] - df_error['Z_Interp']))), 7)
            }

            # rprint(dic)
            dic_errors["RBF Beta"][BETA] = dic_to_print
            # rprint(dic_errors)

            plt_z = ddff.pivot_table(index='X', columns='Y', values='Z').T.values
            X_unique = np.sort(ddff.X.unique())
            Y_unique = np.sort(ddff.Y.unique())
            plt_x, plt_y = np.meshgrid(X_unique, Y_unique)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ct = plt.contourf(plt_x, plt_y, plt_z, cmap='plasma')
            cs = plt.contour(plt_x, plt_y, plt_z, levels=[0])
            ax.clabel(cs, fontsize=8)
            plt.axis('scaled')
            plt.colorbar(ct)
            plt.title(f'RBF Beta {BETA} Simulate')
            plt.savefig(f'{point_folder}Plot_RBF_Beta_{BETA}.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ct = plt.contourf(plt_x, plt_y, plt_z, cmap='plasma')
            cs = plt.contour(plt_x, plt_y, plt_z, levels=[0])
            ax.clabel(cs, fontsize=8)
            plt.scatter(df_drop['X'], df_drop['Y'], s=12, c='red')
            plt.axis('scaled')
            plt.colorbar(ct)
            plt.title(f'RBF Beta {BETA} Simulate vs Validate')
            plt.savefig(f'{point_folder}Plot_RBF_Beta_{BETA}_simulate_vs_validate.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

            # Fig SubPlot Errors
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(cm_to_inch(40), cm_to_inch(20)))
            ct = ax1.contour(plt_x, plt_y, plt_z, colors='k')
            sct = ax1.scatter(df_error['X_Real'], df_error['Y_Real'], s=12,
                              c=df_error['Z_Error_Absolute'], cmap='plasma')  # 'jet') # 'YlOrRd')
            ax1.clabel(ct, fontsize=8)
            ax1.title.set_text('Absolute Error')
            ax1.axis('scaled')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            col_bar = plt.colorbar(sct, ax=ax1, cax=cax1)  # shrink=0.8)
            col_bar.ax.get_yaxis().labelpad = 15
            col_bar.ax.set_ylabel('  meters  ')
            ct = ax2.contour(plt_x, plt_y, plt_z, colors='k')
            sct = ax2.scatter(df_error['X_Real'], df_error['Y_Real'], s=12,
                              c=df_error['Z_Error_Relative'], cmap='plasma')  # 'jet') # 'YlOrRd')
            ax2.clabel(ct, fontsize=8)
            ax2.title.set_text('Relative Error')
            ax2.axis('scaled')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.1)
            col_bar = plt.colorbar(sct, ax=ax2, cax=cax2)  # shrink=0.8)
            col_bar.ax.get_yaxis().labelpad = 15
            col_bar.ax.set_ylabel('  %  ', rotation=270)
            fig.suptitle(f'Errors Distribution of RBF Beta = {BETA}', fontsize=16)
            plt.savefig(f'{point_folder}Plot_Errors_Distribution_RBF_Beta_{BETA}.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X, Y, Z = axes3d.get_test_data(0.05)
            ct3d = ax.contour(plt_x, plt_y, plt_z, levels=[0])
            ax.clabel(ct3d, fontsize=9, inline=1)
            ax.scatter(df_error['X_Real'], df_error['Y_Real'], df_error['Z_Interp'],
                       label='Interpolate Values')
            ax.scatter(df_error['X_Real'], df_error['Y_Real'], df_error['Z_Real'],
                       label='Real Values')
            plt.legend(loc="best")
            plt.title(f'3D Compare Real - Interpolate RBF Beta = {BETA}')
            plt.savefig(f'{point_folder}Plot_Errors_3D_Compare_RBF_Beta_{BETA}.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

        rprint(dic_errors)
