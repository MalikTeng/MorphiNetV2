#include "cuda_geometry.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

/**
 * CUDA kernel for parallel signed distance field computation.
 * Each thread computes the SDF value for one voxel center.
 */
__global__ void compute_sdf_kernel(
    const float* __restrict__ vertices,     // [V, 3] mesh vertices
    const int* __restrict__ faces,          // [F, 3] triangle face indices
    const float* __restrict__ query_points, // [N, 3] voxel center coordinates
    float* __restrict__ sdf_output,         // [N] output signed distances
    int num_vertices,
    int num_faces,
    int num_points
) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;
    
    // Load query point
    float3 query = make_float3(
        query_points[point_idx * 3],
        query_points[point_idx * 3 + 1],
        query_points[point_idx * 3 + 2]
    );
    
    float min_dist_sq = FLT_MAX;
    int closest_face_idx = 0;
    
    // Find closest distance to any triangle face
    for (int face_idx = 0; face_idx < num_faces; face_idx++) {
        // Get triangle vertex indices
        int v0_idx = faces[face_idx * 3];
        int v1_idx = faces[face_idx * 3 + 1];
        int v2_idx = faces[face_idx * 3 + 2];
        
        // Load triangle vertices
        float3 v0 = make_float3_from_ptr(vertices, v0_idx);
        float3 v1 = make_float3_from_ptr(vertices, v1_idx);
        float3 v2 = make_float3_from_ptr(vertices, v2_idx);
        
        // Compute distance to this triangle
        float dist_sq = point_triangle_distance_squared(query, v0, v1, v2);
        
        // Track closest face
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_face_idx = face_idx;
        }
    }
    
    // Compute sign based on closest face normal orientation
    float sign = compute_sign_from_closest_face(query, vertices, faces, closest_face_idx);
    
    // Store signed distance (Trimesh convention: positive = inside)
    sdf_output[point_idx] = sign * sqrtf(min_dist_sq);
}

/**
 * Host function to launch the CUDA kernel with optimal configuration.
 */
extern "C" void launch_sdf_kernel(
    const float* vertices,
    const int* faces,
    const float* grid_points,
    float* sdf_output,
    int num_vertices,
    int num_faces,
    int num_points
) {
    // Determine optimal kernel launch configuration
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Use 256 threads per block (good balance for most GPUs)
    const int threads_per_block = 256;
    const int blocks = (num_points + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    compute_sdf_kernel<<<blocks, threads_per_block>>>(
        vertices, faces, grid_points, sdf_output,
        num_vertices, num_faces, num_points
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();
    
    // Check for kernel execution errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

/**
 * Optimized kernel for processing multiple meshes in a batch.
 * Each block processes one mesh, threads within block process voxel points.
 */
__global__ void compute_sdf_batched_kernel(
    const float* __restrict__ vertices,     // [B*max_V, 3] all vertices
    const int* __restrict__ faces,          // [B*max_F, 3] all faces
    const int* __restrict__ vertex_offsets, // [B+1] cumulative vertex counts
    const int* __restrict__ face_offsets,   // [B+1] cumulative face counts
    const float* __restrict__ query_points, // [N, 3] voxel centers (shared)
    float* __restrict__ sdf_output,         // [B, N] output distances
    int batch_size,
    int num_points
) {
    int batch_idx = blockIdx.x;
    int point_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || point_idx >= num_points) return;
    
    // Get mesh boundaries for this batch
    int vertex_start = vertex_offsets[batch_idx];
    int vertex_end = vertex_offsets[batch_idx + 1];
    int face_start = face_offsets[batch_idx];
    int face_end = face_offsets[batch_idx + 1];
    
    int num_mesh_vertices = vertex_end - vertex_start;
    int num_mesh_faces = face_end - face_start;
    
    if (num_mesh_faces == 0) {
        // Empty mesh - mark as outside
        sdf_output[batch_idx * num_points + point_idx] = 1.0f;
        return;
    }
    
    // Load query point
    float3 query = make_float3(
        query_points[point_idx * 3],
        query_points[point_idx * 3 + 1],
        query_points[point_idx * 3 + 2]
    );
    
    float min_dist_sq = FLT_MAX;
    int closest_face_idx = face_start;
    
    // Process faces for this mesh
    for (int face_idx = face_start; face_idx < face_end; face_idx++) {
        // Get triangle vertices (adjust indices for batch offset)
        int v0_idx = faces[face_idx * 3] + vertex_start;
        int v1_idx = faces[face_idx * 3 + 1] + vertex_start;
        int v2_idx = faces[face_idx * 3 + 2] + vertex_start;
        
        float3 v0 = make_float3_from_ptr(vertices, v0_idx);
        float3 v1 = make_float3_from_ptr(vertices, v1_idx);
        float3 v2 = make_float3_from_ptr(vertices, v2_idx);
        
        float dist_sq = point_triangle_distance_squared(query, v0, v1, v2);
        
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            closest_face_idx = face_idx;
        }
    }
    
    // Compute sign
    float sign = compute_sign_from_closest_face(
        query, vertices, faces, closest_face_idx
    );
    
    // Store result
    sdf_output[batch_idx * num_points + point_idx] = sign * sqrtf(min_dist_sq);
}

/**
 * Host function for batched processing (future optimization).
 */
extern "C" void launch_sdf_batched_kernel(
    const float* vertices,
    const int* faces,
    const int* vertex_offsets,
    const int* face_offsets,
    const float* grid_points,
    float* sdf_output,
    int batch_size,
    int num_points
) {
    // Launch configuration for batched processing
    dim3 block_size(256, 1, 1);
    dim3 grid_size(batch_size, (num_points + 255) / 256, 1);
    
    compute_sdf_batched_kernel<<<grid_size, block_size>>>(
        vertices, faces, vertex_offsets, face_offsets,
        grid_points, sdf_output, batch_size, num_points
    );
    
    cudaDeviceSynchronize();
}