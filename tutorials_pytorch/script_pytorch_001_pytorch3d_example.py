import torch
import pytorch3d.transforms as transforms3d
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer
from pytorch3d.structures import Meshes
import matplotlib.pyplot as plt

# cube vertices
vertices = torch.tensor([
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]
], dtype=torch.float32)

faces = torch.tensor([
    [0, 2, 3],
    [0, 3, 1],
    [4, 6, 7],
    [4, 7, 5],
    [0, 4, 5],
    [0, 5, 1],
    [2, 6, 7],
    [2, 7, 3],
    [0, 4, 6],
    [0, 6, 2],
    [1, 5, 7],
    [1, 7, 3]
], dtype=torch.int64)

# Create a Meshes object
cube_mesh = Meshes(verts=[vertices], faces=[faces])

# Set up a perspective camera
R, T = transforms3d.look_at_view_transform(2.7, 0, 0)  # Camera rotation and translation
cameras = PerspectiveCameras(
    R=R,
    T=T,
    device=torch.device("gpu")  # You can change this to "cpu" if you don't have a GPU
)

# Define rasterization settings
raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)

# Create a mesh renderer
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=torch.device("gpu")  # You can change this to "cpu" if you don't have a GPU
)

# Render the cube
images = renderer(meshes=cube_mesh)

# Display the rendered image
plt.figure(figsize=(5, 5))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()

