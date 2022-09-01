def create_voxelized_splits(input_train_hdf_path, input_val_hdf_path, input_test_hdf_path, output_train_hdf_path, output_val_hdf_path, output_test_hdf_path, num_splits, voxels_per_dim=48, box_dim_angstroms=48):
    
    """
    input:
    1) path/to/input/training/hdf/file.hdf -- NOTE: These should all be the "base" names, without associated numbering (which will be automated)
    2) path/to/input/validation/hdf/file.hdf
    3) path/to/input/testing/hdf/file.hdf
    4) path/to/output/training/hdf/file.hdf
    5) path/to/output/validation/hdf/file.hdf
    6) path/to/output/testing/hdf/file.hdf
    7) number of voxels in an edge of the bounding box. Default is 48
    8) length (in Angstroms) of an edge of the bounding box. Default is 48

    output:
    1) Voxelizes a given number of numerically-labeled splits, each consisting of 3 hdf files containing voxelized training, validation, and testing data
    """

    # necessary import statements
    import h5py
    import numpy as np
    import math

    #define a function to voxelize an array
    def voxelize_one_vox_per_atom(xyz_array, feat, vol_dim, size_angstrom=48):

            # initialize volume
            vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

            #get bounding box
            xmid = (min(xyz_array[:, 0]) + max(xyz_array[:, 0])) / 2
            ymid = (min(xyz_array[:, 1]) + max(xyz_array[:, 1])) / 2
            zmid = (min(xyz_array[:, 2]) + max(xyz_array[:, 2])) / 2
            xmin = xmid - (size_angstrom / 2)
            ymin = ymid - (size_angstrom / 2)
            zmin = zmid - (size_angstrom / 2)
            xmax = xmid + (size_angstrom / 2)
            ymax = ymid + (size_angstrom / 2)
            zmax = zmid + (size_angstrom / 2)

            # assign each atom to voxels
            for ind in range(xyz_array.shape[0]):
                x = xyz_array[ind, 0]
                y = xyz_array[ind, 1]
                z = xyz_array[ind, 2]
                if x < xmin or x >= xmax or y < ymin or y >= ymax or z < zmin or z >= zmax:
                    continue

                cx = math.floor((x - xmin) / (xmax - xmin) * (vol_dim[0]))
                cy = math.floor((y - ymin) / (ymax - ymin) * (vol_dim[1]))
                cz = math.floor((z - zmin) / (zmax - zmin) * (vol_dim[2]))

                vol_data[cx, cy, cz, :] += feat[ind, :]
        
            return vol_data

    #define a function to intake hdf files, voxelize them, and produce output hdf files 
    def voxelize_hdf (input_train_hdf_path, input_val_hdf_path, input_test_hdf_path, output_train_hdf_path, output_val_hdf_path, output_test_hdf_path, voxels_per_dim=48, box_dim_angstroms=48):
        

        # open input hdf files
        input_train_hdf = h5py.File(input_train_hdf_path, 'r')
        input_val_hdf = h5py.File(input_val_hdf_path, 'r')
        input_test_hdf = h5py.File(input_test_hdf_path, 'r')

        # create output hdf files
        output_train_hdf = h5py.File(output_train_hdf_path, 'w')
        output_val_hdf = h5py.File(output_val_hdf_path, 'w')
        output_test_hdf = h5py.File(output_test_hdf_path, 'w')

        # organize input and output hdf files
        input_hdfs = [input_train_hdf, input_val_hdf, input_test_hdf]
        output_hdfs = [output_train_hdf, output_val_hdf, output_test_hdf]


        for input_hdf, output_hdf in zip(input_hdfs, output_hdfs): 
            for pdbid in input_hdf.keys():

                #read input data
                input_data = input_hdf[pdbid]
                input_affinity = input_hdf[pdbid].attrs['affinity']

                #perform voxelization
                output_3d_data = voxelize_one_vox_per_atom(input_data[:, 0:3], input_data[:, 3:], [voxels_per_dim, voxels_per_dim, voxels_per_dim, input_data.shape[1]-3], box_dim_angstroms)
                
                #write output data
                output_hdf.create_dataset(pdbid, data=output_3d_data, shape=output_3d_data.shape, dtype='float32', compression='lzf')
                output_hdf[pdbid].attrs['affinity'] = input_affinity
    
    #perform the actual voxelizations
    for i in range(0, num_splits):
        voxelize_hdf(input_train_hdf_path[:-4] + str(i) + '.hdf', input_val_hdf_path[:-4] + str(i) + '.hdf', input_test_hdf_path[:-4] + str(i) + '.hdf', 
                    output_train_hdf_path[:-4] + str(i) + '.hdf', output_val_hdf_path[:-4] + str(i) + '.hdf', output_test_hdf_path[:-4] + str(i) + '.hdf')
