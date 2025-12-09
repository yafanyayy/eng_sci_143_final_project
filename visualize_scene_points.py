import numpy as np
import plotly.graph_objects as go

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

  # Add cameras
  for i, (R, t) in enumerate(camera_poses):
      # compute camera matrix from rotation and translation
      P = np.hstack((R, t))
      add_plotly_camera(img_h, img_w, P, i, fig)

  fig.update_layout(
      scene=dict(
          aspectmode='data',
          xaxis=dict(title='x'),
          yaxis=dict(title='y'),
          zaxis=dict(title='z')
      ),
      margin=dict(l=0, r=0, t=30, b=0),
      title=f'point cloud (colored by {color_by})',
      scene_camera=dict(
          up=dict(x=0, y=-1, z=0),
          center=dict(x=0, y=0, z=0),
          eye=dict(x=0, y=0, z=-2)
      )
  )

  return fig


