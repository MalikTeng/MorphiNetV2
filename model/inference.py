import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelInference:
    """Handles model inference operations including padding and sliding window inference."""
    
    def __init__(self, super_params):
        """
        Initialize model inference utilities.
        
        Args:
            super_params: Configuration parameters containing layers, upscale_ratio, etc.
        """
        self.super_params = super_params
        self._resnet_padding_logged = False
    
    def _apply_resnet_padding(self, tensor):
        """
        Apply padding to ensure ResNet compatibility with skip connections.
        
        Args:
            tensor: Input tensor with shape (B, C, H, W, D)
        
        Returns:
            Tuple of (padded_tensor, pad_info) where pad_info contains padding information
        """
        raise NotImplementedError("This method is deprecated. Use the new inference_with_padding method instead.")
        # Calculate stride factor based on ResNet layers
        num_layers = len(self.super_params.layers)
        stride_factor = 2 ** (num_layers - 1)  # Encoder downsampling factor
        
        original_shape = tensor.shape
        b, c, h, w, d = original_shape
        
        # Calculate padding needed for each spatial dimension
        pad_h = (stride_factor - h % stride_factor) % stride_factor
        pad_w = (stride_factor - w % stride_factor) % stride_factor
        pad_d = (stride_factor - d % stride_factor) % stride_factor
        
        # Apply padding if needed
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            # PyTorch pad format: (D_left, D_right, W_left, W_right, H_left, H_right)
            padding = (0, pad_d, 0, pad_w, 0, pad_h)
            padded_tensor = F.pad(tensor, padding, mode="constant", value=0)
            
            # Store padding info for removal later
            pad_info = {
                'original_shape': original_shape,
                'pad_h': pad_h,
                'pad_w': pad_w,
                'pad_d': pad_d,
                'upscale_ratio': self.super_params.upscale_ratio
            }
            
            # Log padding info only once
            if not self._resnet_padding_logged:
                self._resnet_padding_logged = True
        else:
            padded_tensor = tensor
            pad_info = {
                'original_shape': original_shape, 
                'pad_h': 0, 
                'pad_w': 0, 
                'pad_d': 0, 
                'upscale_ratio': self.super_params.upscale_ratio
            }
        
        return padded_tensor, pad_info
    
    def _remove_resnet_padding(self, tensor, pad_info):
        """
        Remove padding applied for ResNet compatibility and handle upscaling.
        
        Args:
            tensor: Padded and upscaled tensor from decoder
            pad_info: Dictionary containing padding information
        
        Returns:
            Tensor with padding removed and potentially downscaled
        """
        raise NotImplementedError("This method is deprecated. Use the new inference_with_padding method instead.")
        # Extract padding information
        original_shape = pad_info['original_shape']
        pad_h = pad_info['pad_h']
        pad_w = pad_info['pad_w']
        pad_d = pad_info['pad_d']
        upscale_ratio = pad_info['upscale_ratio']
        
        # Calculate expected upscaled dimensions
        orig_h, orig_w, orig_d = original_shape[2:]
        expected_upscaled_h = (orig_h + pad_h) * upscale_ratio
        expected_upscaled_w = (orig_w + pad_w) * upscale_ratio
        expected_upscaled_d = (orig_d + pad_d) * upscale_ratio
        
        # Remove padding by calculating original spatial dimensions after upscaling
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            # Calculate how much to crop from upscaled tensor
            crop_h = pad_h * upscale_ratio
            crop_w = pad_w * upscale_ratio
            crop_d = pad_d * upscale_ratio
            
            # Crop the padded regions
            if crop_h > 0:
                tensor = tensor[:, :, :-crop_h, :, :]
            if crop_w > 0:
                tensor = tensor[:, :, :, :-crop_w, :]
            if crop_d > 0:
                tensor = tensor[:, :, :, :, :-crop_d]
        
        return tensor
    
    def sliding_window_inference_wrapper(self, model, inputs, roi_size, sw_batch_size=1, predictor=None, mode="gaussian", device=None):
        """
        Wrapper for sliding window inference with proper device handling.
        
        Args:
            model: The neural network model
            inputs: Input tensor
            roi_size: Size of the sliding window
            sw_batch_size: Batch size for sliding window
            predictor: Optional predictor function
            mode: Sliding window mode
            device: Target device
        
        Returns:
            Model predictions
        """
        if device is None:
            device = DEVICE
        
        # Ensure inputs are on the correct device
        if hasattr(inputs, 'to'):
            inputs = inputs.to(device)
        
        # Use MONAI's sliding window inference
        return sliding_window_inference(
            inputs=inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=predictor or model,
            overlap=0.5,
            mode=mode,
            device=device
        )
    
    def inference_with_padding(self, model, inputs, roi_size=None, sw_batch_size=1):
        """
        Perform inference with automatic padding for ResNet compatibility.
        
        Args:
            model: The neural network model
            inputs: Input tensor
            roi_size: Size of the sliding window (optional)
            sw_batch_size: Batch size for sliding window
        
        Returns:
            Model predictions with padding removed
        """
        raise NotImplementedError("This method is deprecated. Use the new inference_with_padding method instead.")
        # Apply padding
        padded_inputs, pad_info = self._apply_resnet_padding(inputs)
        
        # Perform inference
        if roi_size is not None:
            # Use sliding window inference
            predictions = self.sliding_window_inference_wrapper(
                model=model,
                inputs=padded_inputs,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size
            )
        else:
            # Direct inference
            predictions = model(padded_inputs)
        
        # Remove padding
        predictions = self._remove_resnet_padding(predictions, pad_info)
        
        return predictions
    
    def _convert_to_onehot(self, tensor, num_classes, is_prediction=True):
        """
        Convert segmentation tensor to one-hot encoding.
        
        Args:
            tensor: Input tensor
            num_classes: Number of classes
            is_prediction: Whether this is a prediction tensor (affects processing)
        
        Returns:
            One-hot encoded tensor
        """
        if is_prediction:
            # For predictions, apply argmax first to get class indices
            if tensor.dim() == 5:  # (B, C, H, W, D) - 3D case
                tensor = torch.argmax(tensor, dim=1, keepdim=True)
            elif tensor.dim() == 4:  # Could be (B, C, H, W) for 2D MR slices or (C, H, W, D) for 3D
                # Check if this looks like batched 2D slices (batch_size > 1, multiple classes)
                if tensor.shape[1] > 1:  # (B, C, H, W) - batched 2D slices with multiple classes
                    tensor = torch.argmax(tensor, dim=1, keepdim=True)
                else:  # (C, H, W, D) - single 3D volume with multiple channels
                    tensor = torch.argmax(tensor, dim=0, keepdim=True)
        
        # Prepare tensor for one-hot encoding (remove singleton channel dimension)
        if tensor.dim() == 5:  # (B, 1, H, W, D)
            tensor = tensor.squeeze(1)  # (B, H, W, D)
        elif tensor.dim() == 4:
            # Check if we have a singleton batch or channel dimension
            if tensor.shape[1] == 1:  # (B, 1, H, W) - batched 2D with singleton channel
                tensor = tensor.squeeze(1)  # (B, H, W)
            elif tensor.shape[0] == 1:  # (1, H, W, D) - singleton batch
                tensor = tensor.squeeze(0)  # (H, W, D)
        
        # Convert to long tensor for one-hot encoding
        tensor = tensor.long()
        
        # Create one-hot encoding
        one_hot = torch.nn.functional.one_hot(tensor, num_classes=num_classes)
        
        # Rearrange dimensions to put channel dimension first
        if one_hot.dim() == 5:  # (B, H, W, D, C) -> (B, C, H, W, D)
            one_hot = one_hot.permute(0, 4, 1, 2, 3)
        elif one_hot.dim() == 4:  # (B, H, W, C) -> (B, C, H, W) for 2D batched
            one_hot = one_hot.permute(0, 3, 1, 2)
        else:
            raise ValueError("2D slices are not supported for prediction")
        # elif one_hot.dim() == 3:  # (H, W, C) -> (C, H, W) for 2D single  
        #     one_hot = one_hot.permute(2, 0, 1)
        
        return one_hot.float()
    
    def preprocess_for_resnet(self, tensor):
        """
        Preprocess tensor for ResNet input (apply padding).
        
        Args:
            tensor: Input tensor
        
        Returns:
            Tuple of (preprocessed_tensor, pad_info)
        """
        return self._apply_resnet_padding(tensor)
    
    def postprocess_from_resnet(self, tensor, pad_info):
        """
        Postprocess tensor from ResNet output (remove padding).
        
        Args:
            tensor: Output tensor from ResNet
            pad_info: Padding information from preprocessing
        
        Returns:
            Tensor with padding removed
        """
        return self._remove_resnet_padding(tensor, pad_info)