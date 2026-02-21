from typing import Optional, Tuple, Any
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


from sam2.modeling.sam2_base import SAM2Base

import torch
from sam2.build_sam import build_sam2
import os
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import onnx

class SAM2ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base) -> None:
        super().__init__()
        self.model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed

    def forward(self, x: torch.Tensor):
        backbone_out=self.image_encoder(x)
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
                        backbone_out["backbone_fpn"][0]
                    )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        feature_maps = backbone_out["backbone_fpn"][-self.model.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.model.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        feats = [feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
                  for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])][::-1]

        return feats[0],feats[1],feats[2]


class SAM2ImageDecoder(nn.Module):
    def __init__(
            self,
            sam_model: SAM2Base,
            multimask_output: bool
    ) -> None:
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.img_size = sam_model.image_size
        self.multimask_output = multimask_output

    @torch.no_grad()
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
        orig_im_size: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        self.sparse_embedding = sparse_embedding
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        high_res_feats = [high_res_feats_0, high_res_feats_1]
        image_embed = image_embed

        masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        masks = masks[:, 1:, :, :]
        iou_predictions = iou_predictions[:, 1:]

        orig_im_size = orig_im_size.to(torch.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        upscaled_masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        masks = torch.clamp(masks, -32.0, 32.0)
        # return upscaled_masks, iou_predictions, masks
        return upscaled_masks[:1,:1,...], iou_predictions[:1,:1,...], masks[:1,:1,...]

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:

        point_coords = point_coords + 0.5

        point_coords[:, :, 0] = point_coords[:, :, 0] / self.model.image_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.model.image_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.prompt_encoder.not_a_point_embed.weight * (
                point_labels == -1
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.prompt_encoder.point_embeddings[i].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding
    

def get_model_cfg(model_type):
    if model_type == "sam2_hiera_tiny":
        model_cfg = "sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small":
        model_cfg = "sam2_hiera_s.yaml"
    elif model_type == "sam2_hiera_base_plus":
        model_cfg = "sam2_hiera_b+.yaml"
    else:
        model_cfg = "sam2_hiera_l.yaml"
    return model_cfg

def export_encoder(model_type):
    
    curdir = os.path.dirname(os.path.abspath(__file__))
    sam2_checkpoint =  os.path.join(curdir, f"checkpoints/{model_type}.pt")

    sam2_model = build_sam2(get_model_cfg(model_type), sam2_checkpoint, device="cpu")

    img=torch.randn(1, 3, input_size, input_size).cpu()

    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    high_res_feats_0, high_res_feats_1, image_embed = sam2_encoder(img)
    print(high_res_feats_0.shape)
    print(high_res_feats_1.shape)
    print(image_embed.shape)

    torch.onnx.export(sam2_encoder,
                    img,
                    os.path.join(curdir, f"checkpoints/{model_type}_encoder.onnx"),
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names = ['image'],
                    output_names = ['high_res_feats_0', 'high_res_feats_1', 'image_embed']
                    )

def export_decoder(model_type, quantize=False):

    curdir = os.path.dirname(os.path.abspath(__file__))
    sam2_checkpoint = os.path.join(curdir, f"checkpoints/{model_type}.pt")

    sam2_model = build_sam2(get_model_cfg(model_type), sam2_checkpoint, device="cpu")
        

    sam2_encoder = SAM2ImageEncoder(sam2_model).cpu()
    sam2_decoder = SAM2ImageDecoder(sam2_model, multimask_output=multimask_output).cpu()

    img=torch.randn(1, 3, input_size, input_size).cpu()
    high_res_feats_0, high_res_feats_1, image_embed = sam2_encoder(img)

    embed_dim = sam2_model.sam_prompt_encoder.embed_dim
    embed_size = (sam2_model.image_size // sam2_model.backbone_stride, sam2_model.image_size // sam2_model.backbone_stride)
    mask_input_size = [4 * x for x in embed_size]
    print(embed_dim, embed_size, mask_input_size)

    point_coords = torch.randint(low=0, high=input_size, size=(1, 5, 2), dtype=torch.float)
    point_labels = torch.randint(low=0, high=1, size=(1, 5), dtype=torch.float)
    mask_input = torch.randn(1, 1, *mask_input_size, dtype=torch.float)
    has_mask_input = torch.tensor([1], dtype=torch.float)
    orig_im_size = torch.tensor([input_size, input_size], dtype=torch.float)

    masks, scores, low_res_masks = sam2_decoder(image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input, orig_im_size)


    onnx_model_path = os.path.join(curdir, f"checkpoints/{model_type}_decoder.onnx")
    torch.onnx.export(sam2_decoder,
                    (image_embed, high_res_feats_0, high_res_feats_1, point_coords, point_labels, mask_input, has_mask_input, orig_im_size),
                    onnx_model_path,
                    export_params=True,
                    opset_version=17,
                    do_constant_folding=True,
                    input_names = ['image_embed', 'high_res_feats_0', 'high_res_feats_1', 'point_coords', 'point_labels', 'mask_input', 'has_mask_input', 'orig_im_size'],
                    output_names = ['masks', 'iou_predictions', 'low_res_masks'],
                    dynamic_axes = {"point_coords": {0: "num_labels", 1: "num_points"},
                                    "point_labels": {0: "num_labels", 1: "num_points"},
                                    "mask_input": {0: "num_labels"},
                                    "has_mask_input": {0: "num_labels"},
                    }
                    )
    if quantize:
        onnx_model_quantized_path = os.path.join(curdir, f"checkpoints/{model_type}_decoder_quantized.onnx")
        quantize_model(onnx_model_path, onnx_model_quantized_path)
    
def quantize_model(onnx_model_path, onnx_model_quantized_path):
    onnx_model = onnx.load(onnx_model_path)
    quantize_dynamic(
        onnx_model,
        model_output=onnx_model_quantized_path,
        # optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
        nodes_to_exclude=['/Transpose'],
    )
    # onnx_model_path = onnx_model_quantized_path

if __name__ == '__main__':
    # model_type = 'sam2_hiera_base_plus' #@param ["sam2_hiera_tiny", "sam2_hiera_small", "sam2_hiera_large", "sam2_hiera_base_plus"]
    model_type = "sam2_hiera_base_plus"
    # input_size = 768 #@param {type:"slider", min:160, max:4102, step:8}
    input_size = 1024 # Bad output if anything else (for now)
    multimask_output = False #@param {type:"boolean"}

    if model_type == "sam2_hiera_tiny":
        model_cfg = "sam2_hiera_t.yaml"
    elif model_type == "sam2_hiera_small":
        model_cfg = "sam2_hiera_s.yaml"
    elif model_type == "sam2_hiera_base_plus":
        model_cfg = "sam2_hiera_b+.yaml"
    else:
        model_cfg = "sam2_hiera_l.yaml"

    # export_encoder(model_type)
    export_decoder(model_type, quantize=True)
