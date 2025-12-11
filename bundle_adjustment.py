'''
Notes:
https://demuc.de/tutorials/cvpr2017/sparse-modeling.pdf
https://pytorch3d.org/tutorials/bundle_adjustment

Note: need good enough poses to start with or this will just be bad b/c get stuck at local minima

All 3D points X + camera poses P should be learnable parameters (with some limitations, see the planning doc)

Reference my thesis/other pytorch code

loss = MSE loss (the one in the reference is missing the square but it's equivalent b/c squaring doesn't do anything to the relative losses)

optim + gradients + other stuff
'''

import torch
import cv2
import numpy as np

# Project 3D points to 2D using camera parameters and 3D points for ONE camera
def project_points(scene_points, camera_rotation, camera_translation, camera_matrix, distCoeffs):
    # using cv2.projectPoints
    # projected_points = []
    # projected_points.append(cv2.projectPoints(scene_points, camera_rotation, camera_translation, camera_matrix, distCoeffs)[0])
    scene_points = scene_points.detach().numpy().astype(np.float32)
    camera_rotation = camera_rotation.astype(np.float32)
    camera_translation = camera_translation.detach().numpy().astype(np.float32)
    camera_matrix = camera_matrix.detach().numpy().astype(np.float32)

    projected_points = cv2.projectPoints(scene_points, camera_rotation, camera_translation, camera_matrix, distCoeffs)[0]
    projected_points = projected_points.squeeze(1) # Eliminate the leading single dimension
    return projected_points

def project_points_np(scene_points, camera_rotation, camera_translation, camera_matrix, distCoeffs):
    # using cv2.projectPoints
    # projected_points = []
    # projected_points.append(cv2.projectPoints(scene_points, camera_rotation, camera_translation, camera_matrix, distCoeffs)[0])

    projected_points = cv2.projectPoints(scene_points, camera_rotation, camera_translation, camera_matrix, distCoeffs)[0]
    projected_points = projected_points.squeeze(1) # Eliminate the leading single dimension
    return projected_points


def rodrigues_torch(rvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (rvec, shape (3,) or (3,1)) to rotation matrix (3x3) in torch.
    Implements Rodrigues formula in pure torch so gradients propagate.
    """
    # ensure shape (3,)
    r = rvec.reshape(3)
    dtype = r.dtype
    device = r.device
    theta = torch.norm(r)
    # use detach().item() to check near-zero without keeping grad
    if theta.detach().abs().item() < 1e-8:
        return torch.eye(3, dtype=dtype, device=device)
    k = r / theta

    # build skew-symmetric K using tensor ops (avoid torch.tensor([...]) with tensors inside)
    K = torch.zeros((3, 3), dtype=dtype, device=device)
    K[0, 1] = -k[2]
    K[0, 2] = k[1]
    K[1, 0] = k[2]
    K[1, 2] = -k[0]
    K[2, 0] = -k[1]
    K[2, 1] = k[0]

    I = torch.eye(3, dtype=dtype, device=device)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def project_points_torch(scene_points: torch.Tensor, rvec: torch.Tensor, tvec: torch.Tensor, camera_matrix: torch.Tensor, distCoeffs=None) -> torch.Tensor:
    """Project 3D points (N,3) to 2D using torch ops (differentiable).
    - rvec: rotation vector (3,) in axis-angle
    - tvec: translation (3,) or (3,1)
    - camera_matrix: 3x3 torch matrix (fx, fy, cx, cy)
    - distCoeffs: distortion coefficients (k1, k2, p1, p2[, k3]); if provided, both radial and tangential distortion are applied.
    Returns: tensor shape (N, 2)
    """
    # rotation matrix
    R = rodrigues_torch(rvec.reshape(3))

    # ensure shapes
    X = scene_points.reshape(-1, 3)
    t = tvec.reshape(3)

    # transform to camera coordinates
    X_cam = (R @ X.t()).t() + t

    # perspective division (normalized coordinates)
    z = X_cam[:, 2]
    x = X_cam[:, 0] / (z + 1e-8)
    y = X_cam[:, 1] / (z + 1e-8)

    # handle distortion coefficients (k1,k2,p1,p2[,k3])
    # Normalize distCoeffs into a torch vector length 5: [k1,k2,p1,p2,k3]
    dtype = camera_matrix.dtype
    device = camera_matrix.device if hasattr(camera_matrix, 'device') else None
    if distCoeffs is None:
        k1 = k2 = p1 = p2 = k3 = torch.tensor(0.0, dtype=dtype, device=device)
    else:
        if isinstance(distCoeffs, torch.Tensor):
            d = distCoeffs.reshape(-1).to(dtype).to(device) if device is not None else distCoeffs.reshape(-1).to(dtype)
        else:
            d = torch.tensor(np.array(distCoeffs).reshape(-1), dtype=dtype, device=device)
        # pad or truncate to 5 entries
        d_full = torch.zeros(5, dtype=dtype, device=device)
        n = min(d.numel(), 5)
        d_full[:n] = d[:n]
        k1, k2, p1, p2, k3 = d_full[0], d_full[1], d_full[2], d_full[3], d_full[4]

    # radial distortion
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * (r2 * r2) + k3 * (r2 * r2 * r2)

    # tangential distortion
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    # focal and principal point
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    u = fx * x_dist + cx
    v = fy * y_dist + cy

    pts = torch.stack([u, v], dim=1)
    return pts

def compute_camera_matrix(focal_length, principal_point_x, principal_point_y):
    camera_matrix = torch.zeros((3, 3), dtype=torch.float32)
    camera_matrix[0, 0] = focal_length
    camera_matrix[1, 1] = focal_length
    camera_matrix[0, 2] = principal_point_x
    camera_matrix[1, 2] = principal_point_y
    camera_matrix[2, 2] = 1.0
    return camera_matrix

def bundle_adjustment(scene_points, camera_rots, camera_translations, camera_matrix, img_points, distCoeffs):
    '''
    Perform bundle adjustment to refine 3D scene points and camera poses.
    Inputs:
    - scene_points: Nx3 numpy array of initial 3D points
    - camera_rots: Mx3x3 numpy array of initial camera rotation matrices
    - camera_translations: Mx3 numpy array of initial camera translations
    - camera_matrix: 3x3 numpy array of initial camera intrinsic matrix
    - img_points: MxNx2 numpy array of observed 2D image points for each camera
    - distCoeffs: distortion coefficients (k1, k2, p1, p2[, k3]) as numpy array
    Outputs:
    - optimized_scene_points: Nx3 numpy array of refined 3D points
    - optimized_camera_rots: Mx3x3 numpy array of refined camera rotation matrices 
    - optimized_camera_translations: Mx3 numpy array of refined camera translations
    - optimized_camera_matrix: 3x3 numpy array of refined camera intrinsic matrix
    '''
    
    
    loss = torch.nn.MSELoss()
    # Ensure img_points is a torch tensor for loss computation
    if not isinstance(img_points, torch.Tensor):
        img_points = torch.tensor(img_points, dtype=torch.float32)
    
    '''
    Make sure things we want to adjust like scene_points, camera_poses, camera_matrices are learnable parameters
    ^ Parameterize each of elements of the camera matrix instead of the whole thing
    Translation vector can be parameterized together
    Rotation - parameterize with quaternions or Rodrigues format
    Don’t want to learn full rotation matrix 
    Might need to make some things static (ex. First camera)
    '''
    # Set up learnable parameters (?)
    # Write a helper function to construct the camera matrix/principal point from parameters
    # Have focal length as it's own parameter 
    # Question: only turn on ability to optimize the camera matrix for the extension
    focal_length = torch.nn.Parameter(torch.tensor(camera_matrix[0,0], dtype=torch.float32), requires_grad=True)
    principal_point_x = torch.nn.Parameter(torch.tensor(camera_matrix[0,2], dtype=torch.float32), requires_grad=True)
    principal_point_y = torch.nn.Parameter(torch.tensor(camera_matrix[1,2], dtype=torch.float32), requires_grad=True)

    # Parameterize 3D points
    scene_points = torch.nn.Parameter(torch.tensor(scene_points, dtype=torch.float32), requires_grad=True)

    # Parameterize translations
    camera_translations = torch.nn.Parameter(torch.tensor(camera_translations, dtype=torch.float32), requires_grad=True)

    # Convert camera_rotations to axis-angle (Rodrigues) form before parameterizing
    camera_rotations = np.zeros((camera_rots.shape[0], 3, 1), dtype=np.float32)
    num_cameras = camera_rots.shape[0]
    for i in range(num_cameras):
        camera_rotations[i] = cv2.Rodrigues(camera_rots[i])[0].astype(np.float32)
    camera_rotations = torch.nn.Parameter(torch.tensor(camera_rotations, dtype=torch.float32), requires_grad=True)

    # TODO: not sure where to put this? - want to fix first camera pose
    camera_mask = torch.ones(camera_rots.shape[0], 1, dtype=torch.float32)
    camera_mask[0] = 0. # Fix the first camera pose

    # TODO: Change learning rate as needed
    optimizer = torch.optim.Adam([scene_points, camera_rotations, camera_translations, focal_length, principal_point_x, principal_point_y], lr=1e-6)
    # TODO: Increase num_iterations after we verify this works
    num_iterations = 1000

    best_loss = float('inf')
    best_scene_points = None
    best_camera_rotations = None
    best_camera_translations = None
    best_focal_length = None
    best_principal_point_x = None
    best_principal_point_y = None

    for iter in range(num_iterations):
        optimizer.zero_grad()

        # Recompute camera matrix from parameters (keep as tensor so graph connects to focal/principal params)
        camera_matrix = compute_camera_matrix(focal_length, principal_point_x, principal_point_y)

        # Keep loss as a torch tensor so we can call backward() on it
        current_loss = torch.tensor(0.0, device=scene_points.device)
        # Compute loss over all cameras (that are not the first one)
        for i in range(camera_rotations.shape[0]):
            # Use a differentiable torch projection so gradients flow
            rvec = camera_rotations[i].reshape(3)
            tvec = camera_translations[i].reshape(3)
            projected_points = project_points_torch(scene_points, rvec, tvec, camera_matrix, distCoeffs)

            # Ensure img_points[i] is a torch tensor (img_points converted above)
            img_i = img_points[i]

            # Accumulate the tensor loss (don't call .item() here, that detaches the value)
            current_loss = current_loss + loss(projected_points, img_i) * camera_mask[i]

        current_loss.backward()

        # Prevent updates to the first camera (index 0).
        # The mask only zeros the loss contribution but does not stop gradients.
        # Zero the gradients for the fixed camera parameters so optimizer.step()
        # will not change them.
        if hasattr(camera_rotations, 'grad') and camera_rotations.grad is not None:
            try:
                camera_rotations.grad[0].zero_()
            except Exception:
                # Fallback: if shapes differ, set using slice
                camera_rotations.grad[0:1].zero_()

        if hasattr(camera_translations, 'grad') and camera_translations.grad is not None:
            try:
                camera_translations.grad[0].zero_()
            except Exception:
                camera_translations.grad[0:1].zero_()

        optimizer.step()

        # Keep the parameters that correspond to the best loss so far
        if current_loss.item() < best_loss:
            best_loss = current_loss.item()
            best_scene_points = scene_points.detach().clone()
            best_camera_rotations = camera_rotations.detach().clone()
            best_camera_translations = camera_translations.detach().clone()
            best_focal_length = focal_length.detach().clone()
            best_principal_point_x = principal_point_x.detach().clone()
            best_principal_point_y = principal_point_y.detach().clone()

        # Print status report
        if iter % 100 == 0:
            # current_loss is a tensor; use .item() for printing only
            print(f"Iteration {iter}, Loss: {current_loss.item()}")
    
    print("Final loss after bundle adjustment:", current_loss.item())

    print("Best loss after bundle adjustment:", best_loss)
    # Convert all parameters back to numpy arrays for output
    scene_points = best_scene_points.detach().numpy()
    # Convert camera rotations back to rotation matrices
    camera_rots = np.zeros((best_camera_rotations.shape[0], 3, 3), dtype=np.float32)
    for i in range(num_cameras):
        camera_rots[i] = cv2.Rodrigues(best_camera_rotations[i].detach().numpy())[0]
    camera_translations = best_camera_translations.detach().numpy()
    camera_matrix = np.zeros((3,3), dtype=np.float32)
    camera_matrix[0, 0] = best_focal_length.item()
    camera_matrix[1, 1] = best_focal_length.item()
    camera_matrix[0, 2] = best_principal_point_x.item()
    camera_matrix[1, 2] = best_principal_point_y.item()
    camera_matrix[2, 2] = 1.0

    return scene_points, camera_rots, camera_translations, camera_matrix

if __name__ == "__main__":
    # Example usage (with dummy data) using NumPy for test data generation
    num_points = 100
    num_cameras = 5

    # Random 3D points (NumPy)
    scene_points_np = np.random.randn(num_points, 3).astype(np.float32)

    # Random camera rotations (as rotation matrices) and translations (NumPy)
    camera_rotations_np = np.zeros((num_cameras, 3, 3), dtype=np.float32)
    for i in range(num_cameras):
        camera_rotations_np[i] = cv2.Rodrigues(np.random.randn(3))[0].astype(np.float32)

    camera_translations_np = np.random.randn(num_cameras, 3).astype(np.float32)

    # Initial camera matrix (NumPy)
    camera_matrix = np.array([[800, 0, 320],
                              [0, 800, 240],
                              [0, 0, 1]], dtype=np.float32)

    # Dummy image points (simulate with true projected points + noise) using NumPy
    img_points_np = np.zeros((num_cameras, num_points, 2), dtype=np.float32)
    for i in range(num_cameras):
        img_points_np[i] = project_points_np(scene_points_np, camera_rotations_np[i], camera_translations_np[i], camera_matrix, np.zeros((4,1), dtype=np.float32))
        img_points_np[i] += 0.01 * np.random.randn(*img_points_np[i].shape).astype(np.float32) # add small noise 
    print(img_points_np.shape)

    # Dummy distortion coefficients (NumPy)
    distCoeffs = np.zeros((4, 1), dtype=np.float32)

    # Convert NumPy test data to torch tensors before calling bundle_adjustment
    # scene_points_t = torch.from_numpy(scene_points_np).float()
    # camera_rotations_t = torch.from_numpy(camera_rotations_np).float()
    # camera_translations_t = torch.from_numpy(camera_translations_np).float()
    # img_points_t = torch.from_numpy(img_points_np).float()

    # Run bundle adjustment (function expects numpy arrays)
    optimized_scene_points, optimized_camera_rotations, optimized_camera_translations, optimized_camera_matrix = bundle_adjustment(
        scene_points_np,
        camera_rotations_np,
        camera_translations_np,
        camera_matrix,
        img_points_np,
        distCoeffs
    )