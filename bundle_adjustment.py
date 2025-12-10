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
    projected_points = []
    projected_points.append(cv2.projectPoints(scene_points, camera_rotation, camera_translation, camera_matrix, distCoeffs)[0])
    return torch.stack(projected_points, dim=0)

def compute_camera_matrix(focal_length, principal_point_x, principal_point_y):
    camera_matrix = torch.zeros((3, 3), dtype=torch.float32)
    camera_matrix[0, 0] = focal_length
    camera_matrix[1, 1] = focal_length
    camera_matrix[0, 2] = principal_point_x
    camera_matrix[1, 2] = principal_point_y
    camera_matrix[2, 2] = 1.0
    return camera_matrix

def bundle_adjustment(scene_points, camera_rotations, camera_translations, camera_matrix, img_points, distCoeffs):
    loss = torch.nn.MSELoss()
    
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
    focal_length = torch.nn.Parameter(torch.tensor([camera_matrix[0,0]], dtype=torch.float32), requires_grad=True)
    principal_point_x = torch.nn.Parameter(torch.tensor([camera_matrix[0,2]], dtype=torch.float32), requires_grad=True)
    principal_point_y = torch.nn.Parameter(torch.tensor([camera_matrix[1,2]], dtype=torch.float32), requires_grad=True)

    # Paramaterize 3D points
    scene_points = torch.nn.Parameter(scene_points, requires_grad=True)

    # Paramaterize translations
    camera_translations = torch.nn.Parameter(camera_translations, requires_grad=True)

    # Convert camera_rotations to axis-angle (Rodrigues) form before parameterizing
    camera_rotations = cv2.Rodrigues(camera_rotations.detach().numpy())[0]
    camera_rotations = torch.nn.Parameter(torch.tensor(camera_rotations, dtype=torch.float32), requires_grad=True)

    # TODO: not sure where to put this?
    camera_mask = torch.ones(camera_rotations.shape[0], 1, dtype=torch.float32)
    camera_mask[0] = 0. # Fix the first camera pose

    # TODO: fix this parameter passing in depending on how the parameters are set up
    optimizer = torch.optim.Adam([scene_points, camera_rotations, camera_translations, focal_length, principal_point_x, principal_point_y], lr=1e-3)
    # TODO: Increase num_iterations after we verify this works
    num_iterations = 100


    print("Initial loss before bundle adjustment:", loss(project_points(scene_points, camera_rotations, camera_translations, camera_matrix, distCoeffs), img_points).item())

    for iter in range(num_iterations):
        optimizer.zero_grad()

        # Recompute camera matrix from parameters
        camera_matrix = compute_camera_matrix(focal_length, principal_point_x, principal_point_y)

        current_loss = 0.0
        # Compute loss over all cameras (that are not the first one)
        for i in range(1, camera_rotations.shape[0]):
            # Compute matrix form of camera rotation
            rotation_matrix = cv2.Rodrigues(camera_rotations[i].detach().numpy())[0]
            # Compute projected points and loss
            projected_points = project_points(scene_points, rotation_matrix, camera_translations[i], camera_matrix, distCoeffs)
            current_loss += loss(projected_points, img_points[i]) * camera_mask[i]

        current_loss.backward()
        optimizer.step()

        # Print status report
        if iter % 100 == 0:
            print(f"Iteration {iter}, Loss: {current_loss.item()}")
    
    print("Final loss after bundle adjustment:", current_loss.item())

    # Convert all parameters back to numpy arrays for output
    scene_points = scene_points.detach().numpy()
    camera_rotations = cv2.Rodrigues(camera_rotations.detach().numpy())[0]
    camera_translations = camera_translations.detach().numpy()
    camera_matrix = compute_camera_matrix(focal_length.item(), principal_point_x.item(), principal_point_y.item()).detach().numpy()

    return scene_points, camera_rotations, camera_translations, camera_matrix