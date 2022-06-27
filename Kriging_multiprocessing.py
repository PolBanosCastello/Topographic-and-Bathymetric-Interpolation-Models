import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from rich import print as rprint
import numpy as np
import skgstat as skg
import os
from tqdm.auto import tqdm
import math


def cm_to_inch(value):
    return value/2.54


plt.rcParams["figure.figsize"] = [cm_to_inch(40), cm_to_inch(20)]


class Kriging:
    point_folder = f'./Points/Subset_Vilanova/'
    if not os.path.exists(point_folder):
        os.mkdir(point_folder)
    else:
        pass

    def __init__(self, data_frame, df_validation, resolution_factor=50, reduction_scale=1):
        self.points = data_frame
        self.reduction_scale = reduction_scale
        self.df_validation = df_validation

        self.num_points = len(self.points)
        self.resolution_factor = resolution_factor

        self.df_validation.to_csv(f'{self.point_folder}df_validation_{self.num_points}_points.csv',
                                  index=False, header=False)
        self.points.to_csv(f'{self.point_folder}df_points_{self.num_points}_points.csv',
                            index=False, header=False)

        self.V_df = skg.Variogram(self.points[['X', 'Y']].values,
                                  self.points.Z.values, maxlag='median', normalize=False)

    def semivariogram_plot(self, show_plot=True):

        fig, (ax_1, ax_2, ax_3) = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
        x = np.linspace(0, self.V_df.maxlag, 100)
        manual_lags = (6, 12, 18)
        col_lab = ['10 lags', 'varying lags', 'Scott rule']
        # plot each variogram

        self.V_df.bin_func = 'even'
        self.V_df.n_lags = 10
        ax_1.plot(self.V_df.bins, self.V_df.experimental, '.b')
        ax_1.grid(which='major', axis='x')
        ax_1.set_title('10 lags')

        self.V_df.n_lags = 20
        ax_2.plot(self.V_df.bins, self.V_df.experimental, '.b')
        ax_2.grid(which='major', axis='x')
        ax_2.set_title('20 lags')

        self.V_df.bin_func = 'scott'
        ax_3.set_xlabel('Lag (-)')
        ax_3.plot(self.V_df.bins, self.V_df.experimental, '.b')
        ax_3.grid(which='major', axis='x')
        ax_3.set_title('Scott rule lags')

        plt.tight_layout()
        plt.savefig(f'{self.point_folder}Plot_Kriging_SemiVariogramas_1.0.png')
        if show_plot:
            plt.show()
        plt.close()

    def semivariogram_plot_errors(self, show_plot=True):

        self.V_df.bin_func = 'scott'
        fig, _a = plt.subplots(2, 3, sharex=False, sharey=False)
        axes = _a.flatten()
        for i, model in enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic')):
            self.V_df.model = model
            self.V_df.plot(axes=axes[i], hist=False, show=False, grid=False)
            axes[i].set_title('Model: %s; RMSE: %.2f' % (model, self.V_df.rmse))
            axes[i].set_ylim(0, max(self.V_df.experimental))
        plt.savefig(f'{self.point_folder}Plot_Kriging_SemiVariogramas.png')
        if show_plot:
            plt.show()
        plt.close()

    def interpolate_df(self, V, ax, df):
        self.points['X'] = self.points['X'] * self.reduction_scale
        self.points['Y'] = self.points['Y'] * self.reduction_scale

        self.df_validation['X'] = self.df_validation['X'] * self.reduction_scale
        self.df_validation['Y'] = self.df_validation['Y'] * self.reduction_scale

        x_min, x_max = int(np.round(min(df['X']))), int(np.round(max(df['X'])))
        y_min, y_max = int(np.round(min(df['Y']))), int(np.round(max(df['Y'])))

        num_resolution = ((x_max - x_min) * self.resolution_factor)
        x = np.linspace(x_min, x_max, num=int(num_resolution))
        y = np.linspace(y_min, y_max, num=int(num_resolution))

        for a in range(0, len(self.df_validation['X'])):

            x = np.append(x, self.df_validation['X'][a])
            y = np.append(y, self.df_validation['Y'][a])

        self.points['X'] = self.points['X'] / self.reduction_scale
        self.points['Y'] = self.points['Y'] / self.reduction_scale

        self.df_validation['X'] = self.df_validation['X'] / self.reduction_scale
        self.df_validation['Y'] = self.df_validation['Y'] / self.reduction_scale

        x = x / self.reduction_scale
        y = y / self.reduction_scale

        rprint(f'Min-Max X:  {round(min(x), 2)}  -  {round(max(x), 4)}')
        rprint(f'Min-Max Y:  {round(min(y), 2)}  -  {round(max(y), 4)}')
        rprint(f'Size X: {x.size}   -   Size Y: {y.size}')

        xx, yy = np.meshgrid(x, y)
        zz = np.empty(xx.shape)
        zz[:] = np.nan

        ok = skg.OrdinaryKriging(V, min_points=3, max_points=15, mode='exact')
        zz = ok.transform(xx.flatten(), yy.flatten()).reshape(xx.shape)

        art = ax.matshow(zz, origin='lower', cmap='plasma', vmin=V.values.min(), vmax=V.values.max())
        ax.set_title('%s model' % V.model.__name__)
        plt.colorbar(art, ax=ax)
        # plt.show()

        return xx, yy, zz

    def execute_method(self):

        self.fields = []
        fig, _a = plt.subplots(2, 3, figsize=(12, 10), sharex=True, sharey=True)
        axes = _a.flatten()
        for i, model in tqdm(enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic'))):
            self.V_df.model = model
            self.xx, self.yy, zz = self.interpolate_df(self.V_df, axes[i], self.points)
            self.fields.append(zz)

        plt.show()

        self.df_fields = pd.DataFrame(
            {'spherical': self.fields[0].flatten(), 'exponential': self.fields[1].flatten(),
             'gaussian': self.fields[2].flatten(), 'matern': self.fields[3].flatten(),
             'stable': self.fields[4].flatten(), 'cubic': self.fields[5].flatten()}).describe()

        rprint(self.df_fields)

        self.df_fields.to_csv(f'{self.point_folder}df_fields_{self.num_points}_points.csv',
                              index=True, header=True)

        return self.df_fields

    def save_method(self, name):

        for i, model in enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic')):
            zz = self.fields[i]

            list_p = []
            for a in range(0, zz.shape[0]):
                for b in range(0, zz.shape[1]):
                    P = [self.xx[a, b], self.yy[a, b], zz[a, b]]
                    list_p.append(P)
            df_save = pd.DataFrame(list_p)
            df_save.columns = ['X', 'Y', 'Z']
            df_save.to_csv(f'{self.point_folder}kriging_{model}_{name}_{self.num_points}_points.csv',
                           index=False, header=False)

    def load_last_method(self, model, name):

        df_load = pd.read_csv(f'{self.point_folder}kriging_{model}_{name}_{self.num_points}_points.csv', header=None)
        df_load.columns = ['X', 'Y', 'Z']

        return df_load

    def calculate_error(self, df_model, name_save, method='MSE', show_plot=False):

        def find_nearest(df_predict, value_x, value_y, value_z, min_z_validate):
            global x, y
            df_predict_x_array = df_predict.X.to_numpy()
            df_predict_y_array = df_predict.Y.to_numpy()

            idx_x = (min(np.abs(df_predict_x_array - value_x)))
            idx_y = (min(np.abs(df_predict_y_array - value_y)))

            for i in df_predict_x_array:
                if np.abs(i - value_x) == idx_x:
                    x = i

            for i in df_predict_y_array:
                if np.abs(i - value_y) == idx_y:
                    y = i

            df_aprox = df_predict[(df_predict['X'] == x) & (df_predict['Y'] == y)]
            array_aprox = df_aprox.to_numpy()

            escal_z = abs(min_z_validate)*1.2
            val_z_escal = (value_z + escal_z)
            int_z_escal = (array_aprox[0][2] + escal_z)
            e_a_z_escal = abs(val_z_escal - int_z_escal)
            e_r_z_escal = (e_a_z_escal/abs(val_z_escal))*100

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
+            comparative.append(find_nearest(df_model, self.df_validation.X[i], self.df_validation.Y[i],
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
        col_bar = plt.colorbar(sct)
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

            return ((e_abs_x / len(df_comparative)), (e_abs_y / len(df_comparative)), (e_abs_z / len(df_comparative))), \
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
                    (math.sqrt(e_rel_z / len(df_comparative)))), \
                   df_comparative


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
    RES_FACT = 0.15 # 19 cap creus
    SCALE_FACT = 0.15
    ERROR_METHOD = 'RMSE'
    DO_INTERPOLATE = True
    CALCULATE_ERROR = True
    SHOW_PLOTS = False


    df = pd.read_csv(f'{point_folder_dataset}{FILE_NAME}.csv', sep=';')
    print(len(df))

    plt.scatter(df['X'], df['Y'], c=df['Z'], cmap='plasma')
    plt.title('Input Data Vilanova')
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

    krig_fit = Kriging(df_subset, df_drop, resolution_factor=RES_FACT, reduction_scale=SCALE_FACT)
    krig_fit.semivariogram_plot()
    krig_fit.semivariogram_plot_errors()

    if DO_INTERPOLATE:
        dfs = []
        rprint(' ----------------------   Execute Method   ----------------------')

        rprint(f'Number of Subset points are {len(df_subset)}, '
               f'and the number of Validation points are {len(df_drop)}')

        rprint('  -----   Variables   -----  ')
        rprint('File Name:              ', FILE_NAME)
        rprint('Resolution Factor:      ', RES_FACT)
        rprint('Factor Scale:           ', SCALE_FACT)

        dfs = krig_fit.execute_method()
        krig_fit.save_method(FILE_NAME)

    if CALCULATE_ERROR:
        rprint(' ----------------------  Calculate Errors  ----------------------')
        dic_errors = {
            "Kriging Method": {}
        }

        for i, CONCRETE_MODEL in enumerate(('spherical', 'exponential', 'gaussian', 'matern', 'stable', 'cubic')):
            print(i, CONCRETE_MODEL.capitalize())
            rprint(f'Open {CONCRETE_MODEL.capitalize()} Model')
            ddff = krig_fit.load_last_method(CONCRETE_MODEL, FILE_NAME)

            plt.scatter(ddff['X'], ddff['Y'], c=ddff['Z'], cmap='plasma')
            plt.title(f'Output Data Vilanova Kriging {CONCRETE_MODEL.capitalize()}')
            plt.ylabel('Latitude')
            plt.xlabel('Longitude')
            col_bar = plt.colorbar()
            col_bar.ax.set_ylabel('  Meters  ')
            plt.axis('scaled')
            plt.savefig(f'{point_folder}Plot_Output_Data_Vilanova_Kriging_{CONCRETE_MODEL.capitalize()}.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

            abs_error, rel_error, df_error = krig_fit.calculate_error(ddff, method=ERROR_METHOD,
                                                                      name_save=f'Kriging_{CONCRETE_MODEL.capitalize()}')
            print(f'Error Absolut XYZ {ERROR_METHOD} Kriging {CONCRETE_MODEL.capitalize()}:       ', abs_error[0], abs_error[1],
                  abs_error[2])
            print(f'Error Relatiu XYZ {ERROR_METHOD} Kriging {CONCRETE_MODEL.capitalize()}:       ', rel_error[0], rel_error[1],
                  rel_error[2])

            vec_norm_real = np.linalg.norm(df_error['Z_Real'])
            vec_norm_interpolate = np.linalg.norm(df_error['Z_Interp'])

            print('La Norma del Vector Z_Real:                  ', round(vec_norm_real, 4))
            print('La Norma del Vector Z_Interp:                ', round(vec_norm_interpolate, 4))
            print('Diferencia de Normas Real - Interp:          ', round((vec_norm_real - vec_norm_interpolate), 4))
            print('(Norma Real - Norma Interp)/(Norma Real):    ',
                  round(((vec_norm_real - vec_norm_interpolate) / vec_norm_real), 4))
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
                "(Norma Real - Norma Interp)/(Norma Real)": np.round(((vec_norm_real - vec_norm_interpolate) / vec_norm_real),
                                                                     7),
                "Norma(Vector Reals - Vector Interp)": np.round((np.linalg.norm((df_error['Z_Real'] - df_error['Z_Interp']))),
                                                                7)
            }

            dic_to_print = {
                "Absolute Error": np.round(abs_error[2], 7),
                "Relative Error": np.round(rel_error[2], 7),
                "Norma(Vector Reals - Vector Interp)": np.round(
                    (np.linalg.norm((df_error['Z_Real'] - df_error['Z_Interp']))), 7)
            }

            # rprint(dic)
            dic_errors['Kriging Method'][CONCRETE_MODEL.capitalize()] = dic_to_print

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
            plt.title(f'Kriging {CONCRETE_MODEL.capitalize()} Simulate')
            plt.savefig(f'{point_folder}Plot_Kriging_{CONCRETE_MODEL.capitalize()}.png')
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
            plt.title(f'Kriging {CONCRETE_MODEL.capitalize()}: Simulate vs Validate')
            plt.savefig(f'{point_folder}Plot_Kriging_{CONCRETE_MODEL.capitalize()}_simulate_vs_validate.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

            # Fig SubPlot Errors
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ct = ax1.contour(plt_x, plt_y, plt_z, colors='k')
            sct = ax1.scatter(df_error['X_Real'], df_error['Y_Real'], s=12, c=df_error['Z_Error_Absolute'], cmap='plasma')
            ax1.clabel(ct, fontsize=8)
            ax1.title.set_text('Absolute Error')
            ax1.axis('scaled')
            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="5%", pad=0.1)
            col_bar = plt.colorbar(sct, ax=ax1, cax=cax1)  # shrink=0.8)
            col_bar.ax.get_yaxis().labelpad = 15
            col_bar.ax.set_ylabel('  meters  ')
            ct = ax2.contour(plt_x, plt_y, plt_z, colors='k')
            sct = ax2.scatter(df_error['X_Real'], df_error['Y_Real'], s=12, c=df_error['Z_Error_Relative'], cmap='plasma')
            ax2.clabel(ct, fontsize=8)
            ax2.title.set_text('Relative Error')
            ax2.axis('scaled')
            divider2 = make_axes_locatable(ax2)
            cax2 = divider2.append_axes("right", size="5%", pad=0.1)
            col_bar = plt.colorbar(sct, ax=ax2, cax=cax2)  # shrink=0.8)
            col_bar.ax.get_yaxis().labelpad = 15
            col_bar.ax.set_ylabel('  %  ', rotation=270)
            fig.suptitle(f'Errors Distribution of Kriging {CONCRETE_MODEL.capitalize()}', fontsize=16)
            plt.savefig(f'{point_folder}Plot_Errors_Distribution_Kriging_{CONCRETE_MODEL.capitalize()}.png')
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
            plt.title(f'3D Compare Real - Interpolate Kriging {CONCRETE_MODEL.capitalize()}')
            plt.savefig(f'{point_folder}Plot_Errors_3D_Compare_Kriging_{CONCRETE_MODEL.capitalize()}.png')
            if SHOW_PLOTS:
                plt.show()
            plt.close()

        rprint(dic_errors)
