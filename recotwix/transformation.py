

import numpy as np


def calc_nifti_affine(transformation, fov, res=None, thickness=None):
    """
    transformation: transformation matrix, 4x4
    fov: field of view in mm, 3x1 [x, y, z]
    res: resolution, 3x1 [x, y, z]
    thickness: slice thickness in mm; important for 2D scans with slice gap

    % Orientation information
    %--------------------------------------------------------------------------
    % Axial Analyze voxel co-ordinate system:
    % x increases     right to left
    % y increases posterior to anterior
    % z increases  inferior to superior

    % DICOM patient co-ordinate system (LPS):
    % x increases     right to left
    % y increases  anterior to posterior
    % z increases  inferior to superior

    % Talairach coordinates system (RAS):
    % x increases      left to right
    % y increases posterior to anterior
    % z increases  inferior to superior

    https://github.com/rordenlab/NIfTIspace
    """
    if isinstance(fov, list):
        fov = np.array(fov)

    if res is None:
        res = np.round(fov) # 1mm isotropic        
    if thickness is None:
        thickness = fov[2] / res[2]

    # build affine matrix, according to SPM notation (see spm_dicom_convert.m)
    T = transformation
    T[:,1:3] = -T[:,1:3] # experimentally found that y and z need to be flipped
    PixelSpacing = [fov[0] / res[0], fov[1] / res[1]]
    n_slice = res[2]

    R = T[:,0:2] @ np.diag(PixelSpacing)
    x1 = [1,1,1,1]
    x2 = [1,1,n_slice,1] if n_slice > 1 else [0,0,1,0]
    
    zmax = (fov[2] - thickness) / 2
    if zmax < 0.0001:
        zmax = thickness/2

    y1_c = T @ [0, 0, -zmax, 1]
    y2_c = T @ [0, 0, +zmax, 1]
    
    # SBCS Position Vector points to slice center this must be recalculated for DICOM to point to the upper left corner.
    y1 = y1_c - T[:,0] * fov[0]/2 - T[:,1] * fov[1]/2
    y2 = y2_c - T[:,0] * fov[0]/2 - T[:,1] * fov[1]/2

    DicomToPatient  = np.column_stack((y1, y2, R)) @ np.linalg.inv(np.column_stack((x1, x2, np.eye(4,2))))
    AnalyzeToDicom  = np.row_stack( (np.column_stack( (np.diag([1,-1,1]), [0, (res[1]+1), 0]) ), [0,0,0,1]) ) # Flip voxels in y
    PatientToTal    = np.diag([-1, -1, 1, 1]) # Flip mm coords in x and y directions
    affine          = PatientToTal @ DicomToPatient @ AnalyzeToDicom
    return affine @ np.column_stack((np.eye(4,3), [1,1,1,1])) # this part is implemented in SPM nifti.m



def calc_tranformation_matrix(dcm, position):
    # Forming a transformation matrix by combining the direct cosine matrix (DCM) with the position.
    T = np.row_stack((np.column_stack((dcm, position)), [0,0,0,1]))
    return T


def calc_dcm(norm_sag, norm_cor, norm_tra, inplane_rot_rad):
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

    # if main_orientation == 0:
    #     R1 = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # @ mat // inplane mat
    # elif main_orientation == 1:
    #     R1 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
    # else:
    #     R1 = np.eye(3)
    
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