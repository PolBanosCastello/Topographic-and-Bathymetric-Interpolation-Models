import time

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from rich import print as rprint
import numpy as np
import os
from tqdm.auto import tqdm
import math
from math import radians
from concurrent import futures
from rich.progress import Progress
import multiprocessing as mp


def cm_to_inch(value):
    return value/2.54


plt.rcParams["figure.figsize"] = [cm_to_inch(40), cm_to_inch(20)]


# def __multiprocess_index_handler(index, handler, args):
#     """
#     This function adds the index to the return of the handler function. Useful to sort the results of a
#     multi-threaded operation
#     :param index: index to be returned
#     :param handler: function handler to be called
#     :param args: list with arguments of the function handler
#     :return: tuple with (index, xxx) where xxx is whatever the handler function returned
#     """
#     result = handler(*args)  # call the handler
#     return index, result  # add index to the result
#
#
# def multiprocess(arg_list, handler, max_workers=20, text: str = "progress..."):
#     """
#     Splits a repetitive task into several processes
#     :param arg_list: each element in the list will crate a thread and its contents passed to the handler
#     :param handler: function to be invoked by every thread
#     :param max_workers: Max processes to be launched at once
#     :return: a list with the results (ordered as arg_list)
#     :param text: text to be displayed in the progress bar
#     """
#     index = 0  # thread index
#     ctx = mp.get_context('spawn')
#     with futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
#         processes = []  # empty thread list
#         results = []  # empty list of thread results
#         for args in arg_list:
#             # submit tasks to the executor and append the tasks to the thread list
#             processes.append(executor.submit(__multiprocess_index_handler, index, handler, args))
#             index += 1
#
#         with Progress() as progress:  # Use Progress() to show a nice progress bar
#             task = progress.add_task(text, total=index)
#             for future in futures.as_completed(processes):
#                 future_result = future.result()  # result of the handler
#                 results.append(future_result)
#                 progress.update(task, advance=1)
#
#         # sort the results by the index added by __threadify_index_handler
#         sorted_results = sorted(results, key=lambda a: a[0])
#
#         final_results = []  # create a new array without indexes
#         for result in sorted_results:
#             final_results.append(result[1])
#         return final_results


class IDW:
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
        rprint(f'Size X:     {x.size}    -   Size Y:     {y.size}')

        self.xx, self.yy = np.meshgrid(x, y)
        self.zz = np.empty(self.xx.shape)
        self.zz[:] = np.nan

    def calculate_list_dist(self, xx_ij, yy_ij, type_coordinates, earth_radius=6371):
        list_dist = []
        if type_coordinates == 'GPS':
            for a in range(0, len(self.df['X'])):
                d = earth_radius * np.arccos((np.cos(radians(90-yy_ij)) * np.cos(radians(90-self.df['Y'][a])) +
                                              np.sin(radians(90-yy_ij)) * np.sin(radians(90-self.df['Y'][a])) *
                                              np.cos(radians(xx_ij-self.df['X'][a]))))
                if d == 90:
                    rprint(d)
                if d != 0:
                    b = [self.df['X'][a], self.df['Y'][a], self.df['Z'][a], d]
                    list_dist.append(b)
                    rprint(b)
                input()

        if type_coordinates == 'UTM':
            for a in range(0, len(self.df['X'])):
                if math.dist((self.df['X'][a], self.df['Y'][a]), (xx_ij, yy_ij)) != 0:
                    b = [self.df['X'][a], self.df['Y'][a], self.df['Z'][a],
                         math.dist((self.df['X'][a], self.df['Y'][a]), (xx_ij, yy_ij))]
                    list_dist.append(b)
        df_dist = pd.DataFrame(list_dist, columns=['X', 'Y', 'Z', 'distance'])
        df_dist_by_dist = df_dist.sort_values('distance')
        return df_dist_by_dist

    def execute_method_no_multiprocessing(self,
                                          distance,
                                          file_name,
                                          type_coordinates,
                                          earth_radius=6371,
                                          show_prints=False):

        global dist_0_value
        rprint(f'Execute Inverse Distance Weight Interpolation with Radial Distance  = {distance}')

        rprint(f'Process of IDW method with Radial Distance = {distance} ...')
        for i in tqdm(range(0, self.zz.shape[0])):
            for j in range(0, self.zz.shape[1]):
                if show_prints:
                    print(i, j)
                # t = time.time()
                # rprint('[bold]XX - YY', i, '-', j, self.xx[i, j], self.yy[i, j])
                list_coord_and_dist = []
                list_dist = []

                if type_coordinates == 'GPS':
                    for a in range(0, len(self.df['X'])):
                        d = earth_radius * np.arccos(
                            (np.cos(radians(90 - self.yy[i, j])) * np.cos(radians(90 - self.df['Y'][a])) +
                             np.sin(radians(90 - self.yy[i, j])) * np.sin(radians(90 - self.df['Y'][a])) *
                             np.cos(radians(self.xx[i, j] - self.df['X'][a]))))
                        if d != 0:
                            b = [self.df['X'][a], self.df['Y'][a], self.df['Z'][a], self.xx[i, j], self.yy[i, j], d]
                            list_dist.append(b)

                if type_coordinates == 'UTM':
                    for a in range(0, len(self.df['X'])):
                        if math.dist((self.df['X'][a], self.df['Y'][a]), (self.xx[i, j], self.yy[i, j])) != 0:
                            b = [self.df['X'][a], self.df['Y'][a], self.df['Z'][a],
                                 math.dist((self.df['X'][a], self.df['Y'][a]), (self.xx[i, j], self.yy[i, j]))]
                            list_dist.append(b)
                    # for a in range(0, len(self.df['X'])):
                    #     d = math.dist((self.df['X'][a], self.df['Y'][a]), (self.xx[i, j], self.yy[i, j]))
                    #     if not d == 0:
                    #         if d <= distance:
                    #             b = [self.df['X'][a], self.df['Y'][a], self.df['Z'][a], self.xx[i, j], self.yy[i, j], d]
                    #             list_coord_and_dist.append(b)
                    #             list_dist.append(d)
                    #         else:
                    #             pass
                    #     else:
                    #         if show_prints:
                    #             rprint(f'[red]Distancia 0 en en {self.df["X"][a]}, {self.df["Y"][a]}, {self.df["Z"][a]}')
                    #             time.sleep(1.5)
                    #         dist_0_value = self.df['Z'][a]

                list_dist = sorted(list_dist)
                list_coord_and_dist = sorted(list_coord_and_dist)

                if show_prints:
                    rprint(f'Radi = {distance}:', len(list_dist))
                    rprint(f'Head list distances: {list_dist[:5]}')

                df_dist = pd.DataFrame(list_dist, columns=['X', 'Y', 'Z', 'distance'])
                df_dist_by_dist = pd.DataFrame(df_dist.sort_values('distance').values,
                                               columns=['X', 'Y', 'Z', 'distance'])
                rprint(df_dist_by_dist)
                input()

                if show_prints:
                    rprint(df_dist_by_dist)

                vect_w = []
                vect_z = []

                dist_0 = False
                for a in range(0, len(df_dist_by_dist)):
                    d = (math.dist((self.df['X'][a], self.df['Y'][a]), (self.xx[i, j], self.yy[i, j])) ** 2)

                    if d == 0:
                        dist_0 = True

                    else:

                        w = 1 / (df_dist_by_dist['distance'][a] ** 2)
                        vect_w.append(w)
                        vect_z.append(df_dist_by_dist['Z'][a])

                if dist_0:
                    if show_prints:
                        print(f'[red]Radi = {distance}:', 'Diatance 0', '-->',
                              self.xx[i, j], self.yy[i, j], dist_0_value)
                    self.zz[i, j] = dist_0_value
                    time.sleep(1.5)

                else:

                    if len(vect_w) == 0:
                        if show_prints:
                            rprint(f'[yellow]Radi = {distance}:', 'nan')
                        self.zz[i, j] = float("nan")
                        time.sleep(1.5)

                    else:
                        if show_prints:
                            rprint('Before Normalization: Sum W', sum(vect_w), type(sum(vect_w)))
                        vect_w = (vect_w / sum(vect_w))
                        if show_prints:
                            rprint('After Normalization: Sum W', sum(vect_w), type(sum(vect_w)))
                        vect_wz = np.array(vect_w) * np.array(vect_z)
                        self.zz[i, j] = sum(vect_wz)

                if show_prints:
                    a = 1
                    print(a)

        rprint('Interpolation Finish.\nTransform Mesh into DataFrame')

        list_p = []

        for a in range(0, self.zz.shape[0]):
            for b in range(0, self.zz.shape[1]):
                P = [self.xx[a, b], self.yy[a, b], self.zz[a, b]]
                list_p.append(P)

        self.df_new = pd.DataFrame(list_p, columns=['X', 'Y', 'Z'])
        self.save_method(file_name, dist=distance)

        rprint(f'Finish Radial Distance  = {distance}')

        return self.df_new

    def execute_method(self, distance, file_name, show_prints=False):
        rprint(f'Execute Inverse Distance Weight Interpolation with Radial Distance  = {distance}')

        argument_list = []
        list = np.arange(0, self.max_worker + 1)
        index = 1
        for i in list:
            if i != list[-1]:
                # print(i)
                inici = int(self.zz.shape[0] / self.max_worker) * (index - 1)
                final = int(self.zz.shape[0] / self.max_worker) * index
                # print(inici, final, '\n')
                index += 1
            else:
                # print(i)
                inici = int(self.zz.shape[0] / self.max_worker) * self.max_worker
                final = self.zz.shape[0]
                # print(inici, final, '\n')

            myargs = [distance, file_name, 'UTM', inici, final]
            argument_list.append(myargs)
        # rprint(argument_list)

        rprint(f'Multiprocessing of IDW method with Radial Distance = {distance} ...')

        results = self.multiprocess(argument_list, self.execute_method_multiprocessing, max_workers=self.max_worker)
        # rprint(results)
        list_res = []
        for result in results:
            list_res += result
        # rprint(list_res, len(list_res))

        df_lists = pd.DataFrame(list_res, columns=['Z'])
        df_lists.to_csv(f'{self.point_folder}IDW_optim_method.csv',
                           index=False, header=True)

        zz = np.array(list_res).reshape(self.zz.shape[1], self.zz.shape[0])
        rprint(zz)

        rprint('Interpolation Finish.\nTransform Mesh into DataFrame')

        list_p = []

        for a in range(0, self.zz.shape[0]):
            for b in range(0, self.zz.shape[1]):
                P = [self.xx[a, b], self.yy[a, b], zz[a, b]]
                list_p.append(P)

        self.df_new = pd.DataFrame(list_p, columns=['X', 'Y', 'Z'])
        # self.df_new['X'] = self.df_new['X'] / self.reduction_scale
        # self.df_new['Y'] = self.df_new['Y'] / self.reduction_scale

        self.save_method(file_name, dist=distance)

        rprint(f'Finish Radial Distance  = {distance}')

        return self.df_new

    def execute_method_multiprocessing(self,
                                       distance,
                                       file_name,
                                       type_coordinates,
                                       start,
                                       end,
                                       earth_radius=6371,
                                       show_prints=False):

        rprint(f'[green]Start Multiprocessing Interpolation with Radial Distance {distance} --> {start} - {end} -- {self.zz.shape[0]}')

        list_results = []
        list_results_only = []
        for i in tqdm(range(start, end)):
            for j in range(0, self.zz.shape[1]):
                if show_prints:
                    print(i, j)
                list_res_dist = []
                list_dist = []

                if type_coordinates == 'GPS':
                    for a in range(0, len(self.df['X'])):

                        alfa = (np.cos(radians(90 - self.yy[i, j])) * np.cos(radians(90 - self.df['Y'][a])) +
                                np.sin(radians(90 - self.yy[i, j])) * np.sin(radians(90 - self.df['Y'][a])) *
                                np.cos(radians(self.xx[i, j] - self.df['X'][a])))
                        d = earth_radius * np.arccos(alfa)

                        if d <= distance:
                            b = [self.df['X'][a], self.df['Y'][a], self.df['Z'][a], self.xx[i, j], self.yy[i, j], d]
                            list_dist.append(b)
                            list_res_dist.append(d)

                if type_coordinates == 'UTM':

                    for a in range(0, len(self.df['X'])):
                        d = math.dist((self.df['X'][a], self.df['Y'][a]), (self.xx[i, j], self.yy[i, j]))
                        if d <= distance:
                            b = [self.df['X'][a], self.df['Y'][a], self.df['Z'][a], self.xx[i, j], self.yy[i, j], d]
                            list_res_dist.append(d)
                            list_dist.append(b)

                list_dist = sorted(list_dist)

                # if show_prints:
                #     rprint(f'Radi = {distance}:', len(list_dist))
                #     rprint(f'Head list distances: {list_dist[:5]}')

                df_dist = pd.DataFrame(list_dist, columns=['X', 'Y', 'Z', 'Mesh XX', 'Mesh YY', 'distance'])
                df_dist_by_dist = pd.DataFrame(df_dist.sort_values('distance').values,
                                               columns=['X', 'Y', 'Z', 'Mesh XX', 'Mesh YY', 'distance'])
                # rprint(f'[red]DataFrame Distances')
                # rprint(df_dist_by_dist)
                # if show_prints:
                #     rprint(df_dist_by_dist)

                vect_w = []
                vect_z = []
                dist_0 = False

                for a in range(0, len(df_dist_by_dist)):
                    d = df_dist_by_dist['distance'][a]

                    if d == 0:
                        dist_0 = True
                        if show_prints:
                            rprint(f'[red]Radi = {distance}:', 'Diatance 0', '-->',
                                  self.xx[i, j], self.yy[i, j], df_dist_by_dist['Z'][a])
                        # self.zz[i, j] = dist_0_value
                        list_results.append((i, j, df_dist_by_dist['Z'][a], 'd = 0'))
                        list_results_only.append(df_dist_by_dist['Z'][a])
                    else:
                        if not dist_0:
                            w = 1 / (df_dist_by_dist['distance'][a] ** 2)
                            vect_w.append(w)
                            vect_z.append(df_dist_by_dist['Z'][a])

                if not dist_0:

                    if len(vect_w) == 0:
                        if show_prints:
                            rprint(f'[yellwo]Radi = {distance}:', 'nan')
                        # self.zz[i, j] = float('nan')
                        list_results.append((i, j, 'nan', 'len(vect) = 0'))
                        list_results_only.append('nan')

                        # time.sleep(1.5)

                    else:
                        # if show_prints:
                        #     rprint('Before Normalization: Sum W', sum(vect_w), type(sum(vect_w)))
                        vect_w = (vect_w / sum(vect_w))
                        # if show_prints:
                        #     rprint('After Normalization: Sum W', sum(vect_w), type(sum(vect_w)))
                        vect_wz = np.array(vect_w) * np.array(vect_z)
                        # self.zz[i, j] = sum(vect_wz)
                        if show_prints:
                            rprint(f'[blue]Radi = {distance}:', sum(vect_wz))
                        list_results.append((i, j, sum(vect_wz), 'calcule w'))
                        list_results_only.append(sum(vect_wz))


                if show_prints:
                    a = 1
                    print(a)

        rprint(f'[red]Finish Multiprocessing Interpolation with Radial Distance {distance} --> {start} - {end} -- {self.zz.shape[0]}. {len(list_results)}')

        return list_results_only

    def save_method(self, name, dist):

        self.df_new.to_csv(f'{self.point_folder}IDW_optim_method_{name}_Dist_{dist}_{self.num_points}_points.csv',
                           index=False, header=False)

    def load_last_method(self, name, dist):

        self.df_new = pd.read_csv(
            f'{self.point_folder}IDW_optim_method_{name}_Dist_{dist}_{self.num_points}_points.csv',
            header=None)
        self.df_new.columns = ['X', 'Y', 'Z']

        return self.df_new

    def calculate_error(self, name_save, method='MSE', show_plot=False):

        def find_nearest(df_predict, value_x, value_y, value_z, min_z_validate):
            if value_z == 0:
                value_z = 0.00001

            df_predict_x_array = df_predict.X.to_numpy()
            df_predict_y_array = df_predict.Y.to_numpy()
            # rprint('Func --> df:', df_s_x_array, df_s_y_array)

            # rprint(value_x, value_y)
            idx_x = (min(np.abs(df_predict_x_array - value_x)))
            # print(f'Min x is {idx_x}')
            idx_y = (min(np.abs(df_predict_y_array - value_y)))
            # print(f'Min y is {idx_y}')

            # x = [i for i in df_predict_x_array if np.abs(i - value_x) == idx_x]
            # print(x)
            for x_val in df_predict_x_array:
                if np.abs(x_val - value_x) == idx_x:
                    # print(i, '-->', np.abs(i - value_x), f'------ Min x is {idx_x}')
                    x = x_val
            # print('--------', x, '--------')
            for y_val in df_predict_y_array:
                if np.abs(y_val - value_y) == idx_y:
                    # print(i)
                    y = y_val

            # print(x_n, y_n)#, df_subset[x_n, y_n])
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
                   ((abs(value_x - array_aprox[0][0]) / abs(value_x)) * 100),
                   ((abs(value_y - array_aprox[0][1]) / abs(value_y)) * 100),
                   e_r_z_escal,
                   [value_z, array_aprox[0][2], (abs(value_z - array_aprox[0][2])),
                    ((abs(value_z - array_aprox[0][2]) / abs(value_z)) * 100)]]
            return lst

        comparative = []
        for i in tqdm(range(0, len(self.df_validation.values))):
            # print(i)
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
        # col_bar.ax.get_yaxis().labelpad = 15
        # col_bar.ax.set_ylabel('# of contacts', rotation=270)
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

            return ((e_abs_x / len(df_comparative)), (e_abs_y / len(df_comparative)),
                    (e_abs_z / len(df_comparative))), \
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
    RES_FACT = 0.15  # 19 cap creus
    SCALE_FACT = 0.15
    list_dist = [0.75, 10.0, 15.0] # La Gijon GPS
    # list_dist = [0.50, 1.00, 2.50, 5.00, 10.0, 15.0, 20.0] # La Palma GPS
    list_dist = [225, 275] # La Palma UTM
    ERROR_METHOD = 'RMSE'
    DO_INTERPOLATE = True
    CALCULATE_ERROR = True
    SHOW_PLOTS = False

    df = pd.read_csv(f'{point_folder_dataset}{FILE_NAME}.csv', sep=';')
    # df = pd.read_csv(f'{point_folder_dataset}{FILE_NAME}.csv', sep=',')
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

    fit_IDW = IDW(df_subset, df_drop, resolution_factor=RES_FACT, reduction_scale=SCALE_FACT)

    # ddff = fit_IDW.execute_method_no_multiprocessing(distance=100, file_name=FILE_NAME, type_coordinates='UTM',
    #                                                  show_prints=True)

    if DO_INTERPOLATE:
        dfs = []
        rprint(' ----------------------   Execute Method   ----------------------')

        for NUM_DIST in list_dist:
            rprint(f'Number of Subset points are {len(df_subset)}, '
                   f'and the number of Validation points are {len(df_drop)}')
            rprint()
            rprint('  -----   Variables   ----- ')
            rprint('File Name:              ', FILE_NAME)
            rprint('Resolution Factor:      ', RES_FACT)
            rprint('Factor Scale:           ', SCALE_FACT)
            rprint('Distance to the firsts: ', NUM_DIST)

            df = fit_IDW.execute_method(distance=NUM_DIST, file_name=FILE_NAME)
            dfs.append(df)

        rprint(dfs)

    if CALCULATE_ERROR:
        rprint(' ----------------------  Calculate Errors  ----------------------')
        dic_errors = {
            "IDW Distances": {}
        }

        for NUM_DIST in list_dist:
            rprint('Calculating Errors of Dist', NUM_DIST)

            ddff = fit_IDW.load_last_method(FILE_NAME, dist=NUM_DIST)

            plt.scatter(ddff['X'], ddff['Y'], c=ddff['Z'], cmap='plasma')
            plt.title(f'Output Data Cap de Creus Radial Distance {NUM_DIST}')
            plt.ylabel('Latitude')
            plt.xlabel('Longitude')
            col_bar = plt.colorbar()
            col_bar.ax.set_ylabel('  Meters  ')
            plt.axis('scaled')
            plt.savefig(f'{point_folder}Plot_Output_Data_CapCreus_Radial_Distance_{NUM_DIST}.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

            abs_error, rel_error, df_error = fit_IDW.calculate_error(method=ERROR_METHOD,
                                                                     name_save=f'IDW_Dist_{NUM_DIST}')
            print(f'Error Absolut XYZ {ERROR_METHOD} IDW Dist = {NUM_DIST}:       ', abs_error[0], abs_error[1],
                  abs_error[2])
            print(f'Error Relatiu XYZ {ERROR_METHOD} IDW Dist = {NUM_DIST}:       ', rel_error[0], rel_error[1],
                  rel_error[2])

            vec_norm_real = np.linalg.norm(df_error['Z_Real'])
            vec_norm_interpolate = np.linalg.norm(df_error['Z_Interp'])

            print('La Norma del Vector Z_Real:                ', round(vec_norm_real, 4))
            print('La Norma del Vector Z_Interp:              ', round(vec_norm_interpolate, 4))
            print('Diferencia de Normas Real - Interp:        ', round((vec_norm_real - vec_norm_interpolate), 4))
            print('(Norma Real - Norma Interp)/(Norma Real):  ',
                  round(((vec_norm_real - vec_norm_interpolate)/vec_norm_real), 4))
            print('La Norma(Vector Z Reals - Vector Z Interp:   ',
                  round((np.linalg.norm((df_error['Z_Real'] - df_error['Z_Interp']))), 4))

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
                "(Norma Real - Norma Interp)/(Norma Real)":
                    np.round(((vec_norm_real - vec_norm_interpolate) / vec_norm_real), 7),
                "Norma(Vector Reals - Vector Interp)":
                    np.round((np.linalg.norm((df_error['Z_Real'] - df_error['Z_Interp']))), 7)
            }

            dic_to_print = {
                "Absolute Error": np.round(abs_error[2], 7),
                "Relative Error": np.round(rel_error[2], 7),
                "Norma(Vector Reals - Vector Interp)": np.round(
                    (np.linalg.norm((df_error['Z_Real'] - df_error['Z_Interp']))), 7)
            }

            # rprint(dic)
            dic_errors['IDW Distances'][NUM_DIST] = dic_to_print
            # rprint(dic_errors)

            plt_z = ddff.pivot_table(index='X', columns='Y', values='Z').T.values
            X_unique = np.sort(ddff.X.unique())
            Y_unique = np.sort(ddff.Y.unique())
            plt_x, plt_y = np.meshgrid(X_unique, Y_unique)

            rprint(plt_x.shape)
            rprint(plt_y.shape)
            rprint(plt_z.shape)

            if plt_x.shape == plt_y.shape == plt_z.shape:
                try:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ct = plt.contourf(plt_x, plt_y, plt_z, cmap='plasma')
                    cs = plt.contour(plt_x, plt_y, plt_z, levels=[0])
                    ax.clabel(cs, fontsize=8)
                    plt.axis('scaled')
                    plt.colorbar(ct)
                    plt.title(f'IDW Distances {NUM_DIST} Simulate')
                    plt.savefig(f'{point_folder}Plot_IDW_Distances_{NUM_DIST}.png')
                    if SHOW_PLOTS:
                        plt.show()
                    plt.close()
                except:
                    rprint(f'Error in plot: IDW Distances {NUM_DIST} Simulate')

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ct = plt.contourf(plt_x, plt_y, plt_z, cmap='plasma')
                cs = plt.contour(plt_x, plt_y, plt_z, levels=[0])
                ax.clabel(cs, fontsize=8)
                plt.scatter(df_drop['X'], df_drop['Y'], s=12, c='red')
                plt.axis('scaled')
                plt.colorbar(ct)
                plt.title(f'IDW Distances {NUM_DIST} Simulate vs Validate')
                plt.savefig(f'{point_folder}Plot_IDW_Distances_{NUM_DIST}_simulate_vs_validate.png')
                if SHOW_PLOTS:
                    plt.show()
                plt.close()

                # Fig SubPlot Errors
                fig, (ax1, ax2) = plt.subplots(1, 2)
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
                fig.suptitle(f'Errors Distribution of IDW Distance = {NUM_DIST}', fontsize=16)
                plt.savefig(f'{point_folder}Plot_Errors_Distribution_IDW_Distance_{NUM_DIST}.png')
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
                plt.title(f'3D Compare Real - Interpolate IDW Distance = {NUM_DIST}')
                plt.savefig(f'{point_folder}Plot_Errors_3D_Compare_IDW_Distance_{NUM_DIST}.png')
                if SHOW_PLOTS:
                    plt.show()
                plt.close()

        rprint(dic_errors)
