import torch
import numpy as np
from trimesh import Trimesh, load
from trimesh.convex import convex_hull
from pytorch3d.structures import Meshes
from pytorch3d.ops import taubin_smoothing
from pytorch3d.ops.marching_cubes import marching_cubes
from pytorch3d.io import load_objs_as_meshes
from model.networks import LocalMeshWarper


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MeshOperations:
    """Handles mesh operations for MorphiNet including template mesh processing and surface extraction."""
    
    def __init__(self, super_params, vert_label=None):
        """
        Initialize mesh operations.
        
        Args:
            super_params: Configuration parameters
            vert_label: Vertex labels for template mesh
        """
        self.super_params = super_params
        self.vert_label = vert_label
        self.mesh_c = None
        
        # Initialize local mesh warper
        if hasattr(super_params, 'iteration'):
            self.local_mesh_warper = LocalMeshWarper(super_params.iteration).to(DEVICE)
    
    def _mesh_label(self, mesh):
        """
        Label vertices of template mesh based on color coding.
        
        Args:
            mesh: Template mesh with vertex colors
        
        Sets:
            self.vert_label: Tensor of vertex labels
            self.mesh_c: Center points of LV and RV mesh regions
        """
        COLOR_MAPPING = {
            (1, 0, 0): 0,   # LV-ENDO
            (0, 1, 0): 1,   # RV-ENDO
            (0, 0, 1): 2,   # LV-EPI
            (1, 1, 0): 3,   # RV-EPI
            (1, 0, 1): 4,   # MV & AAV
            (0, 1, 1): 5,   # TV
            (1, 1, 1): 6,   # PV
            (0, 0, 0): 7,   # LEAVE OUT
        }
        
        vert_label = mesh.visual.vertex_colors[:, :3]
        vert_label = np.where(vert_label <= 85, 0, 1)
        vert_label = np.array([COLOR_MAPPING[tuple(c)] for c in vert_label])
        self.vert_label = torch.tensor(vert_label, dtype=torch.long, device=DEVICE)
        
        # Create convex hulls for LV and RV regions
        mesh_lv = convex_hull(mesh.vertices[np.any(np.stack([vert_label == i for i in [0, 2]]), axis=0)])  # LV-ENDO and LV-EPI
        mesh_rv = convex_hull(mesh.vertices[np.any(np.stack([vert_label == i for i in [1, 3]]), axis=0)])  # RV-ENDO and RV-EPI
        
        # Stack centers of both LV and RV
        self.mesh_c = torch.tensor([mesh_lv.center_mass, mesh_rv.center_mass], device=DEVICE)
    
    def surface_extractor(self, seg_true, labels=None):
        """
        Extract surface meshes from segmentation using marching cubes.
        
        Args:
            seg_true: Ground truth segmentation tensor
            labels: Integer or list of integers/lists specifying which labels to extract
                   If None, uses default [[2]] for myocardium
                   If integer, extracts only that label
                   If list, each element can be an integer or list of integers to combine
        
        Returns:
            List of surface meshes with vertices and faces in NDC space [-1, 1]
        """
        # Ensure segmentation tensor is int32 to avoid PyTorch3D compatibility issues
        seg_true = seg_true.to(torch.int32)
        
        # Handle labels parameter
        if labels is None:
            # For GSN phase: extract only myocardium surface as originally designed
            seg_idx_list = [[2]]   # myocardium
        elif isinstance(labels, int):
            # Single label
            seg_idx_list = [[labels]]
        elif isinstance(labels, list):
            # List of labels or lists of labels
            seg_idx_list = []
            for label_group in labels:
                if isinstance(label_group, int):
                    seg_idx_list.append([label_group])
                elif isinstance(label_group, list):
                    seg_idx_list.append(label_group)
                else:
                    raise ValueError(f"Invalid label type in labels list: {type(label_group)}")
        else:
            raise ValueError(f"Invalid labels type: {type(labels)}. Must be None, int, or list.")
        
        # Create binary masks for each label group
        # Note: Use torch.int32 explicitly to avoid PyTorch3D compatibility issues
        seg_true_multi = []
        for seg_idx in seg_idx_list:
            # Create boolean mask and convert to int32
            mask = torch.any(torch.stack([seg_true == i for i in seg_idx]), dim=0)
            seg_true_multi.append(mask)

        mesh_true = []
        for seg_true_ in seg_true_multi:
            # Apply marching cubes to extract surface
            # Move to CPU to avoid PyTorch3D CUDA bug (issue #1679)
            volume_cpu = seg_true_.squeeze(1).to(torch.float32).cpu()
            verts, faces = marching_cubes(
                volume_cpu,
                isolevel=0.5,
                return_local_coords=True,
            )
            
            # Move results back to original device for subsequent operations
            verts = [v.to(seg_true_.device) for v in verts]
            faces = [f.to(seg_true_.device) for f in faces]
            
            # Apply Taubin smoothing to the extracted mesh
            mesh_true.append(taubin_smoothing(Meshes(verts, faces), 0.7, -0.73, 30))

        return mesh_true
    
    @torch.no_grad()
    def warp_template_mesh(self, df_preds):
        """
        Warp template mesh using predicted distance fields.
        
        Args:
            df_preds: Predicted distance field tensor
        
        Returns:
            Warped template mesh with vertices and faces in NDC space
        """
        b, *_, d = df_preds.shape

        def find_rotation_matrix_xz(vector_msh, vector_df):
            """Find rotation matrix to align mesh center with distance field center in XZ plane."""
            # Project vectors onto xz-plane
            vector_msh_xz = torch.stack([vector_msh[:, 0], vector_msh[:, 2]], dim=1)
            vector_df_xz = torch.stack([vector_df[:, 0], vector_df[:, 2]], dim=1)

            # Normalize the projected vectors
            vector_msh_xz = vector_msh_xz / torch.norm(vector_msh_xz, dim=1, keepdim=True)
            vector_df_xz = vector_df_xz / torch.norm(vector_df_xz, dim=1, keepdim=True)

            # Calculate the cosine of the angle between the projected vectors
            cos_theta = torch.sum(vector_msh_xz * vector_df_xz, dim=1)

            # Calculate the sine of the angle using the determinant of 2x2 matrix
            sin_theta = vector_msh_xz[:, 0] * vector_df_xz[:, 1] - vector_msh_xz[:, 1] * vector_df_xz[:, 0]

            # Create rotation matrices
            R = torch.zeros(vector_msh.shape[0], 3, 3, device=vector_msh.device, dtype=torch.float32)
            R[:, 0, 0] = cos_theta
            R[:, 0, 2] = sin_theta
            R[:, 1, 1] = 1
            R[:, 2, 0] = -sin_theta
            R[:, 2, 2] = cos_theta

            return R

        # Load template mesh using PyTorch3D
        template_mesh_pt3d = load_objs_as_meshes([self.super_params.template_mesh_dir], device=DEVICE)
        template_mesh = Meshes(
            verts=[template_mesh_pt3d.verts_packed().to(dtype=torch.float32)], 
            faces=[template_mesh_pt3d.faces_packed().to(dtype=torch.int32)]
        ).to(DEVICE).extend(b)

        # Apply taubin smoothing to improve mesh quality
        template_mesh = taubin_smoothing(template_mesh, 0.5, -0.53, 10)

        # Stage 1: Smooth global offset with rotation alignment
        verts = template_mesh.verts_padded()
        
        # Find the rotation matrix that makes the centroid vectors align
        # Use RV distance field (channel 2) for RV center calculation
        df_c = torch.stack([2 * (torch.nonzero(df <= 1).to(torch.float32).mean(0) / d - 0.5) 
                            for df in df_preds[:, 2]])[:, [2, 1, 0]]   # reorder dimensions, using RV channel
        
        # Use only the RV center as the reference center
        mesh_c = self.mesh_c[1].unsqueeze(0).expand(b, -1).to(torch.float32)
        
        R = find_rotation_matrix_xz(mesh_c, df_c)
        
        # Ensure verts are in double precision before matrix multiplication
        verts = verts.to(torch.float32)
        verts = torch.bmm(R, verts.transpose(1, 2)).transpose(1, 2)
        
        template_mesh = template_mesh.update_padded(verts)

        # Stage 2: Local offset using LocalMeshWarper
        if self.local_mesh_warper is not None:
            template_mesh = self.local_mesh_warper(template_mesh, df_preds, self.vert_label)

        return template_mesh
    
    def initialize_template_mesh(self, template_mesh_path):
        """
        Initialize and label the template mesh.
        
        Args:
            template_mesh_path: Path to template mesh file
        
        Returns:
            Loaded template mesh object (trimesh object for vertex color access)
        """
        # Load with trimesh for vertex color processing (required by _mesh_label)
        template_mesh = load(template_mesh_path)
        self._mesh_label(template_mesh)
        return template_mesh