import numpy as np
import plotly.graph_objects as go
from es143_utils import add_plotly_camera

def visualize_scene_points(scene_points, camera_poses, img_h, img_w):
  # Add points to figure as a scatterplot
  # Clean points slightly
  valid = np.all(np.isfinite(scene_points), axis=2)

  if not np.any(valid):
      print("no valid points to plot.")
      return go.Figure()

  # extract X, Y, Z values for each point
  X = scene_points[:, :, 0][valid]
  Y = scene_points[:, :, 1][valid]
  Z = scene_points[:, :, 2][valid]

  # Plot points and make figure
  fig = go.Figure(
        data=[
            go.Scatter3d(
                x=X, y=Y, z=Z,
                mode='markers',
                marker=dict(size=2,
                            opacity = 0.8)
            )
        ]
    )

    # TODO: depending on what gets input, might need to change this code to accept camara_rotations and camera_translations as two different inputs
  # Add cameras
  for i, (R, t) in enumerate(camera_poses):
      # compute camera matrix from rotation and translation
      P = np.hstack((R, t))
      add_plotly_camera(img_h, img_w, P, 2.0, fig)

  fig.update_layout(
      scene=dict(
          aspectmode='data',
          xaxis=dict(title='x'),
          yaxis=dict(title='y'),
          zaxis=dict(title='z')
      ),
      margin=dict(l=0, r=0, t=30, b=0),
      title=f'point cloud',
      scene_camera=dict(
          up=dict(x=0, y=-1, z=0),
          center=dict(x=0, y=0, z=0),
          eye=dict(x=0, y=0, z=-2)
      )
  )

  return fig


# Test code
if __name__ == "__main__":
    # Generate some random scene points and camera poses for testing
    num_cameras = 5
    num_points = 100
    img_h, img_w = 480, 640

    scene_points = np.random.randn(num_cameras, num_points, 3) * 5.0 + np.array([0, 0, 10])
    camera_poses = []
    for i in range(num_cameras):
        angle = i * (2 * np.pi / num_cameras)
        R = np.array([[np.cos(angle), 0, np.sin(angle)],
                      [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])
        t = np.array([[5 * np.cos(angle)], [0], [5 * np.sin(angle)]])
        camera_poses.append((R, t))

    fig = visualize_scene_points(scene_points, camera_poses, img_h, img_w)
    fig.show()