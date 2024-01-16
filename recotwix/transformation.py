

import numpy as np
from scipy import linalg, ndimage

def calc_nifti_affine(transformation, fov, res, thickness):
    """
    transformation: transformation matrix, 4x4
    fov (dict): field of view in mm, 3x1 [x, y, z]
    res: resolution, 3x1 [x, y, z]
    thickness: slice thickness in mm; slice thickness is not necessary fov['z']/res['z], e.g., 2D scans with slice gap

    following the instructions provided in https://nipy.org/nibabel/coordinate_systems.html and https://nipy.org/nibabel/dicom/dicom_orientation.html
    other useful links: https://www.slicer.org/wiki/Coordinate_systems
    """
    if isinstance(transformation, list):
        transformation = np.array(transformation)

    T = transformation
    if T.shape != (4,4):
        raise ValueError('transformation matrix should be 4x4')
    
    # Here we swap order of x and y because we would like to have PE and RO as the first and the second dimensions, respectively, in nifti file.
    # scaling
    if res['z'] == 1:
        PixelSpacing = [fov['y']/res['y'], fov['x']/res['x'], thickness, 1]
    else:
        PixelSpacing = [fov['y']/res['y'], fov['x']/res['x'], (fov['z']-thickness)/(res['z'] - 1), 1] # to include slice-gap
    scaling_affine = np.zeros([4,4])
    np.fill_diagonal(scaling_affine, PixelSpacing)

    # print(transformation, 'fov=', fov, 'res=', res, 'thickness=', thickness, 'PixelSpacing=', PixelSpacing)
    #rotation
    rotation_affine = T.copy()
    rotation_affine[0:3,-1] = 0

    # translation
    corner_mm = np.array([-fov['y']/2, -fov['x']/2, -(fov['z']-thickness)/2, 1])
    offset = T @ corner_mm
    translation_affine = np.eye(4)
    translation_affine[:,-1] = offset

    # LPS to RAS, Note LPS and PCS (patient coordinate system [Sag, Cor, Tra] ) are identical here 
    PatientToTal = np.diag([-1, -1, 1, 1]) # Flip mm coords in x and y directions

    affine = PatientToTal @ translation_affine @ rotation_affine @ scaling_affine
    return affine



def calc_tranformation_matrix(dcm, position):
    # Forming a transformation matrix by combining the direct cosine matrix (DCM) with the position.
    T = np.row_stack((np.column_stack((dcm, position)), [0,0,0,1]))
    return T


def calc_norm2dcm(norm_sag, norm_cor, norm_tra, inplane_rot_rad):
    # Creating direct cosine matrix (DCM) from plane normal vector and inplane rotation.
    R1 = calc_plane_rotation(norm_sag, norm_cor, norm_tra)
    R2 = calc_inplane_rotation(inplane_rot_rad)
    dcm = np.matmul(R1, R2)
    return dcm


def calc_plane_rotation(norm_sag, norm_cor, norm_tra):
    # reference : https://math.stackexchange.com/questions/180418
    normal = [norm_sag, norm_cor, norm_tra]
    normal = normal / np.linalg.norm(normal)
    main_orientation = np.argmax(np.abs(normal))
    # AA: create a matrix descripts RO_PE_SLC which depends on main orientation
    if main_orientation == 0: # sagittal
        R1 = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    elif main_orientation == 1: # coronal
        R1 = [[1, 0, 0], [0, 0, 1], [0,-1, 0]]
    else: # transversal
        R1 = [[0,-1, 0], [1, 0, 0], [0, 0, 1]]
    
    init_normal = np.zeros(3)
    init_normal[main_orientation] = 1
    # find a rotation that rotate init_normal to normal
    v = np.cross(init_normal, normal) 
    s = np.linalg.norm(v)   # sine of angle
    c = np.dot(init_normal, normal)  # cosine of angle
    if s <= 0.00001:
        R2 = np.eye(3)*c
    else:
        v_x = np.cross(np.eye(3), v)
        R2 = np.eye(3) + v_x + np.divide(np.matmul(v_x, v_x), 1 + c)

    return np.matmul(R2, R1)


def calc_inplane_rotation(inplane_rot_rad):
    r = inplane_rot_rad
    # rot_inplane = rotz(-r)
    rot_inplane =  [[np.cos(-r), np.sin(+r), 0],
                    [np.sin(-r), np.cos(-r), 0],
                    [0         , 0         , 1]]
    return rot_inplane


def resample(volume_data, volume_affine, target_affine, target_size, interp_oder=3, fill_value=0):
    """
    resample volume_data to target_affine and target_size

    volume_data (np.array): 2D, 3D, or 4D data to be resliced 
    volume_affine (np.array): 4x4 transformation matrix of volume_data
    target_affine (np.array): 4x4 transformation matrix of target space
    target_size (list 3x1): size of target space
    interp_oder (int): spline interpolation order
    fill_value (float): value for out of bound voxels
    """

    if volume_data.ndim < 2 or volume_data.ndim > 4:
        raise ValueError('input must be 2D, 3D, or 4D')
    if len(target_size) != 3:
        raise ValueError('target_size must be 3D')
    # ensure input array has 4 dimensions
    while volume_data.ndim < 4:
        volume_data = np.expand_dims(volume_data, axis=-1)

    T = linalg.lstsq(volume_affine, target_affine)[0] # calculate -> inv(ref_affine) @ target_affine
    resampled_vol = []
    for vol_data in np.moveaxis(volume_data, 3, 0):
        resampled_vol.append(ndimage.affine_transform(vol_data, T[0:3,:], output_shape=target_size, order=interp_oder, cval=fill_value))

    resampled_vol = np.moveaxis(np.asarray(resampled_vol), 0, -1).squeeze()
    return resampled_vol


def resample_to_standard_sapce(volume_data, volume_affine, interp_oder=3, fill_value=0):
    fov = 300   # mm
    res = 1.5   # mm
    target_size   = [fov/res]*3
    target_affine = np.hstack((np.eye(4,3)*res, np.array([-fov/2,-fov/2,-fov/2,1]).reshape(4,1)))
    return resample(volume_data, volume_affine, target_affine, target_size, interp_oder, fill_value)