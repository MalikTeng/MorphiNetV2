#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// Declare CUDA kernel launcher (defined in cuda_kernels.cu)
extern "C" void launch_sdf_kernel(
    const float* vertices,
    const int* faces,
    const float* grid_points,
    float* sdf_output,
    int num_vertices,
    int num_faces,
    int num_points
);

extern "C" void launch_sdf_batched_kernel(
    const float* vertices,
    const int* faces,
    const int* vertex_offsets,
    const int* face_offsets,
    const float* grid_points,
    float* sdf_output,
    int batch_size,
    int num_points
);

/**
 * CUDA-accelerated signed distance field computation for mesh voxelization.
 * 
 * Args:
 *   vertices: [B, V, 3] mesh vertices in normalized coordinates [-1, 1]
 *   faces: [B, F, 3] triangle face indices
 *   grid_points: [N, 3] voxel center coordinates in normalized space
 *   shape_dhw: [3] voxel grid dimensions [D, H, W]
 * 
 * Returns:
 *   volume: [B, 1, D, H, W] binary voxel volume (1.0 = inside, 0.0 = outside)
 */
torch::Tensor voxelize_cuda_forward(
    torch::Tensor vertices,      // [B, V, 3] 
    torch::Tensor faces,         // [B, F, 3]
    torch::Tensor grid_points,   // [N, 3]
    torch::Tensor shape_dhw      // [3] - D, H, W as tensor
) {
    // Input validation
    TORCH_CHECK(vertices.device().is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(faces.device().is_cuda(), "faces must be a CUDA tensor");
    TORCH_CHECK(grid_points.device().is_cuda(), "grid_points must be a CUDA tensor");
    TORCH_CHECK(shape_dhw.device().is_cpu(), "shape_dhw should be a CPU tensor");
    
    TORCH_CHECK(vertices.dim() == 3, "vertices must be 3D tensor [B, V, 3]");
    TORCH_CHECK(faces.dim() == 3, "faces must be 3D tensor [B, F, 3]");
    TORCH_CHECK(grid_points.dim() == 2, "grid_points must be 2D tensor [N, 3]");
    TORCH_CHECK(shape_dhw.dim() == 1 && shape_dhw.size(0) == 3, "shape_dhw must be [3]");
    
    TORCH_CHECK(vertices.dtype() == torch::kFloat32, "vertices must be float32");
    TORCH_CHECK(faces.dtype() == torch::kInt32 || faces.dtype() == torch::kInt64, "faces must be int32 or int32");
    TORCH_CHECK(grid_points.dtype() == torch::kFloat32, "grid_points must be float32");
    
    // Convert faces to int32 if needed
    if (faces.dtype() == torch::kInt64) {
        faces = faces.to(torch::kInt32);
    }
    
    // Extract dimensions
    const int batch_size = vertices.size(0);
    const int max_vertices = vertices.size(1);
    const int max_faces = faces.size(1);
    const int num_points = grid_points.size(0);
    
    const int D = shape_dhw[0].item<int>();
    const int H = shape_dhw[1].item<int>();
    const int W = shape_dhw[2].item<int>();
    
    TORCH_CHECK(num_points == D * H * W, 
                "Number of grid points must match D * H * W");
    
    // Allocate output tensor on GPU
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(vertices.device())
        .requires_grad(false);
    
    torch::Tensor volume = torch::zeros({batch_size, 1, D, H, W}, options);
    torch::Tensor sdf_temp = torch::zeros({batch_size, num_points}, options);
    
    // Process each batch item separately for now
    // (Future optimization: use batched kernel)
    for (int b = 0; b < batch_size; b++) {
        // Get current batch data
        torch::Tensor batch_vertices = vertices[b].contiguous();
        torch::Tensor batch_faces = faces[b].contiguous();
        
        // Count actual vertices and faces (skip padding)
        int actual_vertices = max_vertices;
        int actual_faces = max_faces;
        
        // For efficiency, we could implement padding detection here
        // For now, assume all batches have same dimensions
        
        // Launch CUDA kernel for this batch
        launch_sdf_kernel(
            batch_vertices.data_ptr<float>(),
            batch_faces.data_ptr<int>(),
            grid_points.data_ptr<float>(),
            sdf_temp[b].data_ptr<float>(),
            actual_vertices,
            actual_faces,
            num_points
        );
    }
    
    // Convert SDF to binary volume and reshape
    torch::Tensor binary_volume = (sdf_temp > 0.0f).to(torch::kFloat32);
    binary_volume = binary_volume.view({batch_size, 1, D, H, W});
    
    return binary_volume;
}

/**
 * Alternative entry point that works with signed distance field directly.
 * Useful for debugging and applications that need continuous distance values.
 */
torch::Tensor compute_sdf_cuda(
    torch::Tensor vertices,      // [B, V, 3]
    torch::Tensor faces,         // [B, F, 3] 
    torch::Tensor grid_points    // [N, 3]
) {
    // Similar validation as above
    TORCH_CHECK(vertices.device().is_cuda(), "vertices must be a CUDA tensor");
    TORCH_CHECK(faces.device().is_cuda(), "faces must be a CUDA tensor");
    TORCH_CHECK(grid_points.device().is_cuda(), "grid_points must be a CUDA tensor");
    
    // Convert faces to int32 if needed
    if (faces.dtype() == torch::kInt64) {
        faces = faces.to(torch::kInt32);
    }
    
    const int batch_size = vertices.size(0);
    const int max_vertices = vertices.size(1);
    const int max_faces = faces.size(1);
    const int num_points = grid_points.size(0);
    
    // Allocate output for SDF values
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(vertices.device())
        .requires_grad(false);
    
    torch::Tensor sdf_output = torch::zeros({batch_size, num_points}, options);
    
    // Process each batch
    for (int b = 0; b < batch_size; b++) {
        torch::Tensor batch_vertices = vertices[b].contiguous();
        torch::Tensor batch_faces = faces[b].contiguous();
        
        launch_sdf_kernel(
            batch_vertices.data_ptr<float>(),
            batch_faces.data_ptr<int>(),
            grid_points.data_ptr<float>(),
            sdf_output[b].data_ptr<float>(),
            max_vertices,
            max_faces,
            num_points
        );
    }
    
    return sdf_output;
}

/**
 * Utility function to check CUDA availability and device properties.
 */
std::vector<std::string> cuda_info() {
    std::vector<std::string> info;
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    info.push_back("CUDA devices: " + std::to_string(device_count));
    
    if (device_count > 0) {
        int current_device;
        cudaGetDevice(&current_device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, current_device);
        
        info.push_back("Current device: " + std::to_string(current_device));
        info.push_back("Device name: " + std::string(prop.name));
        info.push_back("Compute capability: " + 
                      std::to_string(prop.major) + "." + std::to_string(prop.minor));
        info.push_back("Memory: " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB");
        info.push_back("Max threads per block: " + std::to_string(prop.maxThreadsPerBlock));
    }
    
    return info;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA-accelerated mesh voxelization using signed distance fields";
    
    m.def("voxelize_cuda_forward", &voxelize_cuda_forward, 
          "Fast CUDA voxelization of triangle meshes");
    
    m.def("compute_sdf_cuda", &compute_sdf_cuda,
          "Compute signed distance field using CUDA");
    
    m.def("cuda_info", &cuda_info,
          "Get CUDA device information");
}