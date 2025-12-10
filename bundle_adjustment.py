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

def bundle_adjustment(scene_points, camera_poses, camera_matrix, img_points):
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
    scene_points = torch.nn.Parameter(scene_points)
    camera_poses = torch.nn.Parameter(camera_poses)
    camera_matrix = torch.nn.Parameter(camera_matrix)


    # TODO: not sure where to put this?
    camera_mask = torch.ones(camera_poses.shape[0], 1, dtype=torch.float32)
    camera_mask[0] = 0. # Fix the first camera pose

    # TODO: fix this parameter passing in depending on how the parameters are set up
    optimizer = torch.optim.Adam(list(scene_points.parameters()) + list(camera_poses.parameters()), lr=1e-3)
    num_iterations = 1000

    print("Initial loss before bundle adjustment:", loss(project_points(scene_points, camera_poses, camera_matrix), img_points).item())

    for iter in range(num_iterations):
        optimizer.zero_grad()

        # TODO: change project points to whatever function we have that computes the image points (?)
        projected_points = project_points(scene_points, camera_poses, camera_matrix)

        current_loss = loss(projected_points, img_points)

        current_loss.backward()
        optimizer.step()

        # Print status report
        if iter % 100 == 0:
            print(f"Iteration {iter}, Loss: {current_loss.item()}")
    
    print("Final loss after bundle adjustment:", current_loss.item())
    
    return scene_points, camera_poses, camera_matrix