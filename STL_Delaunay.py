import pandas as pd
import matplotlib.pyplot as plt
from rich import print as rprint
import numpy as np
import pyvista as pv
from tqdm.auto import tqdm
import os
import time


class Structure3D:
    point_folder = './Points/'
    if not os.path.exists(point_folder):
        os.mkdir(point_folder)
    else:
        pass

    def __init__(self, data_frame):
        self.df = data_frame

    def save_method(self, name, method):
        self.df.to_csv(f'{self.point_folder}{method}_DataFrame_{name}.csv', index=False)

    def load_last_method(self, name, method):
        self.df = pd.read_csv(f'{self.point_folder}{method}_DataFrame_{name}.csv')
        self.df.columns = ['X', 'Y', 'Z']
        return self.df

    def blocs_to_print(self):
        t = time.strftime("%Y%m%d-%H%M%S")

        self.df['X'] = self.df['X'] - min(self.df['X'])
        self.df['Y'] = self.df['Y'] - min(self.df['Y'])
        self.df['Z'] = (self.df['Z'] + abs(min(self.df['Z']))) / 10

        rprint('Values of X:    ', min(self.df['X']), '   -   ', max(self.df['X']))
        rprint('Values of Y:    ', min(self.df['Y']), '   -   ', max(self.df['Y']))
        rprint('Values of Z:    ', min(self.df['Z']), '   -   ', max(self.df['Z']))

        limit_in_x = max(self.df['X']) * 1/2
        limit_in_y = max(self.df['Y']) * 1/4

        rprint('Limit inter Bloc in X:', limit_in_x)
        rprint('Limit inter Bloc in Y:', limit_in_y)

        list_B_11 = []
        list_B_12 = []
        list_B_13 = []
        list_B_14 = []
        list_B_21 = []
        list_B_22 = []
        list_B_23 = []
        list_B_24 = []

        for data in self.df.values:

            x = data[0]
            y = data[1]
            z = data[2]

            if x < limit_in_x:
                if y < limit_in_y:
                    list_B_11.append([x, y, z])

                elif limit_in_y <= y < 2*limit_in_y:
                    list_B_12.append([x, y, z])

                elif 2*limit_in_y <= y < 3*limit_in_y:
                    list_B_13.append([x, y, z])

                elif y >= 3*limit_in_y:
                    list_B_14.append([x, y, z])

            elif x >= limit_in_x:
                if y < limit_in_y:
                    list_B_21.append([x, y, z])

                elif limit_in_y <= y < 2 * limit_in_y:
                    list_B_22.append([x, y, z])

                elif 2 * limit_in_y <= y < 3 * limit_in_y:
                    list_B_23.append([x, y, z])

                elif y >= 3 * limit_in_y:
                    list_B_24.append([x, y, z])

        rprint(list_B_11)
        df_B_11 = pd.DataFrame(list_B_11)
        rprint(df_B_11)
        df_B_11.columns = ['X', 'Y', 'Z']
        df_B_11.to_csv(f'{self.point_folder}df_B_11_{t}.csv', index=False)
        rprint('Bloc 1 1 X:   ', min(df_B_11['X']), max(df_B_11['X']))
        rprint('Bloc 1 1 Y:   ', min(df_B_11['Y']), max(df_B_11['Y']))
        rprint()

        df_B_12 = pd.DataFrame(list_B_12)
        df_B_12.columns = ['X', 'Y', 'Z']
        df_B_12.to_csv(f'{self.point_folder}df_B_12_{t}.csv', index=False)
        rprint('Bloc 1 2 X:   ', min(df_B_12['X']), max(df_B_12['X']))
        rprint('Bloc 1 2 Y:   ', min(df_B_12['Y']), max(df_B_12['Y']))
        rprint()

        df_B_13 = pd.DataFrame(list_B_13)
        df_B_13.columns = ['X', 'Y', 'Z']
        df_B_13.to_csv(f'{self.point_folder}df_B_13_{t}.csv', index=False)
        rprint('Bloc 1 3 X:   ', min(df_B_13['X']), max(df_B_13['X']))
        rprint('Bloc 1 3 Y:   ', min(df_B_13['Y']), max(df_B_13['Y']))
        rprint()

        df_B_14 = pd.DataFrame(list_B_14)
        df_B_14.columns = ['X', 'Y', 'Z']
        df_B_14.to_csv(f'{self.point_folder}df_B_14_{t}.csv', index=False)
        rprint('Bloc 1 4 X:   ', min(df_B_14['X']), max(df_B_14['X']))
        rprint('Bloc 1 4 Y:   ', min(df_B_14['Y']), max(df_B_14['Y']))
        rprint()

        df_B_21 = pd.DataFrame(list_B_21)
        df_B_21.columns = ['X', 'Y', 'Z']
        df_B_21.to_csv(f'{self.point_folder}df_B_21_{t}.csv', index=False)
        rprint('Bloc 2 1 X:   ', min(df_B_21['X']), max(df_B_21['X']))
        rprint('Bloc 2 1 Y:   ', min(df_B_21['Y']), max(df_B_21['Y']))
        rprint()

        df_B_22 = pd.DataFrame(list_B_22)
        df_B_22.columns = ['X', 'Y', 'Z']
        df_B_22.to_csv(f'{self.point_folder}df_B_22_{t}.csv', index=False)
        rprint('Bloc 2 2 X:   ', min(df_B_22['X']), max(df_B_22['X']))
        rprint('Bloc 2 2 Y:   ', min(df_B_22['Y']), max(df_B_22['Y']))
        rprint()

        df_B_23 = pd.DataFrame(list_B_23)
        df_B_23.columns = ['X', 'Y', 'Z']
        df_B_23.to_csv(f'{self.point_folder}df_B_23_{t}.csv', index=False)
        rprint('Bloc 2 3 X:   ', min(df_B_23['X']), max(df_B_23['X']))
        rprint('Bloc 2 3 Y:   ', min(df_B_23['Y']), max(df_B_23['Y']))
        rprint()

        df_B_24 = pd.DataFrame(list_B_24)
        df_B_24.columns = ['X', 'Y', 'Z']
        df_B_24.to_csv(f'{self.point_folder}df_B_24_{t}.csv', index=False)
        rprint('Bloc 2 4 X:   ', min(df_B_24['X']), max(df_B_24['X']))
        rprint('Bloc 2 4 Y:   ', min(df_B_24['Y']), max(df_B_24['Y']))
        rprint()

        return [df_B_11, df_B_12, df_B_13, df_B_14, df_B_21, df_B_22, df_B_23, df_B_24], t

    def blocs_to_print_2(self):
        t = time.strftime("%Y%m%d-%H%M%S")

        self.df['X'] = self.df['X'] - min(self.df['X'])

        self.df['Y'] = self.df['Y'] - min(self.df['Y'])

        self.df['Z'] = (self.df['Z'] + abs(min(self.df['Z']))) / 10

        rprint('Values of X:    ', min(self.df['X']), '   -   ', max(self.df['X']))
        rprint('Values of Y:    ', min(self.df['Y']), '   -   ', max(self.df['Y']))
        rprint('Values of Z:    ', min(self.df['Z']), '   -   ', max(self.df['Z']))

        limit_in_x = max(self.df['X']) * 1 / 2
        limit_in_y = max(self.df['Y']) * 1 / 3

        rprint('Limit inter Bloc in X:', limit_in_x)
        rprint('Limit inter Bloc in Y:', limit_in_y)

        list_B_11 = []
        list_B_12 = []
        list_B_13 = []
        # list_B_14 = []
        list_B_21 = []
        list_B_22 = []
        list_B_23 = []
        # list_B_24 = []

        for data in self.df.values:

            x = data[0]
            y = data[1]
            z = data[2]

            if x < limit_in_x:
                if y < limit_in_y:
                    list_B_11.append([x, y, z])

                elif limit_in_y <= y < 2 * limit_in_y:
                    list_B_12.append([x, y, z])

                elif 2 * limit_in_y <= y < 3 * limit_in_y:
                    list_B_13.append([x, y, z])

                # elif y >= 3 * limit_in_y:
                #     list_B_14.append([x, y, z])

            elif x >= limit_in_x:
                if y < limit_in_y:
                    list_B_21.append([x, y, z])

                elif limit_in_y <= y < 2 * limit_in_y:
                    list_B_22.append([x, y, z])

                elif 2 * limit_in_y <= y < 3 * limit_in_y:
                    list_B_23.append([x, y, z])

                # elif y >= 3 * limit_in_y:
                #     list_B_24.append([x, y, z])

        rprint(list_B_11)
        df_B_11 = pd.DataFrame(list_B_11)
        rprint(df_B_11)
        df_B_11.columns = ['X', 'Y', 'Z']
        df_B_11.to_csv(f'{self.point_folder}df_B_11_{t}.csv', index=False)
        rprint('Bloc 1 1 X:   ', min(df_B_11['X']), max(df_B_11['X']))
        rprint('Bloc 1 1 Y:   ', min(df_B_11['Y']), max(df_B_11['Y']))
        rprint()

        df_B_12 = pd.DataFrame(list_B_12)
        df_B_12.columns = ['X', 'Y', 'Z']
        df_B_12.to_csv(f'{self.point_folder}df_B_12_{t}.csv', index=False)
        rprint('Bloc 1 2 X:   ', min(df_B_12['X']), max(df_B_12['X']))
        rprint('Bloc 1 2 Y:   ', min(df_B_12['Y']), max(df_B_12['Y']))
        rprint()

        df_B_13 = pd.DataFrame(list_B_13)
        df_B_13.columns = ['X', 'Y', 'Z']
        df_B_13.to_csv(f'{self.point_folder}df_B_13_{t}.csv', index=False)
        rprint('Bloc 1 3 X:   ', min(df_B_13['X']), max(df_B_13['X']))
        rprint('Bloc 1 3 Y:   ', min(df_B_13['Y']), max(df_B_13['Y']))
        rprint()

        # df_B_14 = pd.DataFrame(list_B_14)
        # df_B_14.columns = ['X', 'Y', 'Z']
        # df_B_14.to_csv(f'{self.point_folder}df_B_14_{t}.csv', index=False)
        # rprint('Bloc 1 4 X:   ', min(df_B_14['X']), max(df_B_14['X']))
        # rprint('Bloc 1 4 Y:   ', min(df_B_14['Y']), max(df_B_14['Y']))
        # rprint()

        df_B_21 = pd.DataFrame(list_B_21)
        df_B_21.columns = ['X', 'Y', 'Z']
        df_B_21.to_csv(f'{self.point_folder}df_B_21_{t}.csv', index=False)
        rprint('Bloc 2 1 X:   ', min(df_B_21['X']), max(df_B_21['X']))
        rprint('Bloc 2 1 Y:   ', min(df_B_21['Y']), max(df_B_21['Y']))
        rprint()

        df_B_22 = pd.DataFrame(list_B_22)
        df_B_22.columns = ['X', 'Y', 'Z']
        df_B_22.to_csv(f'{self.point_folder}df_B_22_{t}.csv', index=False)
        rprint('Bloc 2 2 X:   ', min(df_B_22['X']), max(df_B_22['X']))
        rprint('Bloc 2 2 Y:   ', min(df_B_22['Y']), max(df_B_22['Y']))
        rprint()

        df_B_23 = pd.DataFrame(list_B_23)
        df_B_23.columns = ['X', 'Y', 'Z']
        df_B_23.to_csv(f'{self.point_folder}df_B_23_{t}.csv', index=False)
        rprint('Bloc 2 3 X:   ', min(df_B_23['X']), max(df_B_23['X']))
        rprint('Bloc 2 3 Y:   ', min(df_B_23['Y']), max(df_B_23['Y']))
        rprint()

        return [df_B_11, df_B_12, df_B_13, df_B_21, df_B_22, df_B_23], t

    def make_stl(self, df_block, mesh_name_to_save, alfa_Delaunay_3D=1):
        rprint('Starting to generate a Delaunay STL ...')
        points = df_block.to_numpy()
        cloud = pv.PolyData(points)
        cloud.plot()

        volume = cloud.delaunay_2d()
        shell = volume.extract_geometry()
        shell.save(mesh_name_to_save + '.stl')
        rprint("The Delaunay STL it's done")
        shell.plot()


point_folder = './Points/'
if not os.path.exists(point_folder):
    os.mkdir(point_folder)
else:
    pass

NAME = 'Subset_LaPalma_0.01'
MODEL = 'kriging_spherical'
NUM_POINTS = '2714_points'
STL_TOTAL = True
STL_BLOCS = False

df = pd.read_csv(f'{point_folder}{MODEL}_{NAME}_{NUM_POINTS}.csv', header=None)
df.columns = ['X', 'Y', 'Z']

rprint("File it's open to create a Delaunay STL")

df['Z'] = df['Z'] / 3000

structure_3d_fit = Structure3D(df)
'''STL en Blocs'''
if STL_BLOCS:
    blocks, time = structure_3d_fit.blocs_to_print_2()
    x = 1
    for block in tqdm(blocks):
        # print(x)
        # rprint(block, type(block))
        name = f'{x}'
        structure_3d_fit.make_stl(block, name, alfa_Delaunay_3D=1)
        x += 1

'''STL de tot el Bloc'''
if STL_TOTAL:
    structure_3d_fit.make_stl(df, f'{point_folder}{MODEL}_{NAME}_{NUM_POINTS}')
    # structure_3d_fit.from_df_to_txt(df, MODEL, NAME, NUM_POINTS)
