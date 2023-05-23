import torch
import torch.nn as nn
from torchvision import models

# from efficientnet_pytorch import EfficientNet
import pdb
import timm


def get_my_model(model_name, num_classes):
    model_type = None

    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        model_type = "res"

    if model_name == "resnet18_f":
        model_ft = models.resnet18(pretrained=False)
        model_type = "res"

    if model_name == "resnet34":
        model_ft = models.resnet34(pretrained=True)
        model_type = "res"

    if model_name == "resnet34_f":
        model_ft = models.resnet34(pretrained=False)
        model_type = "res"

    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=True)
        model_type = "res"

    if model_name == "resnet50_f":
        model_ft = models.resnet50(pretrained=False)
        model_type = "res"

    if model_name == "resnet101":
        model_ft = models.resnet101(pretrained=True)
        model_type = "res"

    if model_name == "resnet101_f":
        model_ft = models.resnet101(pretrained=False)
        model_type = "res"

    if model_name == "resnet152":
        model_ft = models.resnet152(pretrained=True)
        model_type = "res"

    if model_name == "resnet152_f":
        model_ft = models.resnet152(pretrained=False)
        model_type = "res"

    if model_name == "wide_resnet50_2":
        model_ft = models.wide_resnet50_2(pretrained=True)
        model_type = "res"

    if model_name == "wide_resnet50_2_f":
        model_ft = models.wide_resnet50_2(pretrained=False)
        model_type = "res"

    if model_name == "wide_resnet101_2":
        model_ft = models.wide_resnet101_2(pretrained=True)
        model_type = "res"

    if model_name == "wide_resnet101_2_f":
        model_ft = models.wide_resnet101_2(pretrained=False)
        model_type = "res"

    if model_name == "resnext50_32x4d":
        model_ft = models.resnext50_32x4d(pretrained=True)
        model_type = "res"

    if model_name == "resnext50_32x4d_f":
        model_ft = models.resnext50_32x4d(pretrained=False)
        model_type = "res"

    if model_name == "resnext101_32x8d":
        model_ft = models.resnext101_32x8d(pretrained=True)
        model_type = "res"

    if model_name == "resnext101_32x8d_f":
        model_ft = models.resnext101_32x8d(pretrained=False)
        model_type = "res"

    if model_name == "eff_b0":
        model_ft = EfficientNet.from_pretrained(
            "efficientnet-b0", num_classes=num_classes
        )
        model_type = "eff"

    if model_name == "eff_b2":
        model_ft = EfficientNet.from_pretrained(
            "efficientnet-b2", num_classes=num_classes
        )
        model_type = "eff"

    if model_name == "eff_b4":
        model_ft = EfficientNet.from_pretrained(
            "efficientnet-b4", num_classes=num_classes
        )
        model_type = "eff"

    if model_name == "eff_b6":
        model_ft = EfficientNet.from_pretrained(
            "efficientnet-b6", num_classes=num_classes
        )
        model_type = "eff"

    if model_name == "eff_b7":
        model_ft = EfficientNet.from_pretrained("efficientnet-b7")
        model_type = "eff"

    if model_name == "densenet121":
        model_ft = models.densenet121(pretrained=True)
        model_type = "den"

    if model_name == "densenet121_f":
        model_ft = models.densenet121(pretrained=False)
        model_type = "den"

    if model_name == "densenet201_f":
        model_ft = models.densenet201(pretrained=False)
        model_type = "den"

    if model_name == "vgg16":
        model_ft = models.vgg16(pretrained=True)
        model_type = "vgg"

    if model_name == "vgg16_f":
        model_ft = models.vgg16(pretrained=False)
        model_type = "vgg"

    if model_name == "mnasnet1_0":
        model_ft = models.mnasnet1_0(pretrained=True)
        model_type = "mnas"

    # timm

    if model_name == "timm_resnet50":
        model_ft = timm.create_model(
            "resnet50", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_resnet50_f":
        model_ft = timm.create_model(
            "resnet50", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_wide_resnet50_2":
        model_ft = timm.create_model(
            "wide_resnet50_2", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_wide_resnet50_2_f":
        model_ft = timm.create_model(
            "wide_resnet50_2", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_resnext50_32x4d":
        model_ft = timm.create_model(
            "resnext50_32x4d", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_resnext50_32x4d_f":
        model_ft = timm.create_model(
            "resnext50_32x4d", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_resnext101_32x8d":
        model_ft = timm.create_model(
            "resnext101_32x8d", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_eff_b0":
        model_ft = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_tf_efficientnetv2_s":
        model_ft = timm.create_model(
            "tf_efficientnetv2_s", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_tf_efficientnetv2_m":
        model_ft = timm.create_model(
            "tf_efficientnetv2_m", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_tf_efficientnetv2_l":
        model_ft = timm.create_model(
            "tf_efficientnetv2_l", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_tf_efficientnetv2_l_f":
        model_ft = timm.create_model(
            "tf_efficientnetv2_l", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_vit_small_patch16_224":
        model_ft = timm.create_model(
            "vit_small_patch16_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_vit_small_patch16_224_f":
        model_ft = timm.create_model(
            "vit_small_patch16_224", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_vit_base_patch16_224":
        model_ft = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_vit_base_patch16_224_f":
        model_ft = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_vit_large_patch16_224":
        model_ft = timm.create_model(
            "vit_large_patch16_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_vit_large_patch16_224_f":
        model_ft = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_vit_base_patch16_224_in21k":
        model_ft = timm.create_model(
            "vit_base_patch16_224_in21k", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_vit_base_patch16_224_miil":
        model_ft = timm.create_model(
            "vit_base_patch16_224_miil", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_vit_base_patch16_224_miil_in21k":
        model_ft = timm.create_model(
            "vit_base_patch16_224_miil_in21k", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_vit_large_r50_s32_224":
        model_ft = timm.create_model(
            "vit_large_r50_s32_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_vit_large_r50_s32_224_f":
        model_ft = timm.create_model(
            "vit_large_r50_s32_224", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_vit_base_patch32_224":
        model_ft = timm.create_model(
            "vit_base_patch32_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_mixer_b16_224":
        model_ft = timm.create_model(
            "mixer_b16_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_mixer_b16_224_in21k":
        model_ft = timm.create_model(
            "mixer_b16_224_in21k", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_mixer_b16_224_miil":
        model_ft = timm.create_model(
            "mixer_b16_224_miil", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_mixer_b16_224_miil_in21k":
        model_ft = timm.create_model(
            "mixer_b16_224_miil_in21k", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_mixer_l16_224":
        model_ft = timm.create_model(
            "mixer_l16_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_gmixer_24_224":
        model_ft = timm.create_model(
            "gmixer_24_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_swin_small_patch4_window7_224":
        model_ft = timm.create_model(
            "swin_small_patch4_window7_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_swin_base_patch4_window7_224":
        model_ft = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_swin_large_patch4_window7_224":
        model_ft = timm.create_model(
            "swin_large_patch4_window7_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_swin_large_patch4_window7_224_f":
        model_ft = timm.create_model(
            "swin_large_patch4_window7_224", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_deit_base_distilled_patch16_224":
        model_ft = timm.create_model(
            "deit_base_distilled_patch16_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_deit_base_patch16_224":
        model_ft = timm.create_model(
            "deit_base_patch16_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_cait_s24_224":
        model_ft = timm.create_model(
            "cait_s24_224", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_convit_base":
        model_ft = timm.create_model(
            "convit_base", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_pit_b_224":
        model_ft = timm.create_model(
            "pit_b_224", pretrained=True, num_classes=num_classes
        )

    # timm preactresnet
    if model_name == "timm_resnetv2_50_f":
        model_ft = timm.create_model(
            "resnetv2_50", pretrained=False, num_classes=num_classes
        )
    if model_name == "timm_resnetv2_101_f":
        model_ft = timm.create_model(
            "resnetv2_101", pretrained=False, num_classes=num_classes
        )

    if model_name == "timm_resnetv2_152_f":
        model_ft = timm.create_model(
            "resnetv2_152", pretrained=False, num_classes=num_classes
        )

    if model_name == "my_resnetv2_50_nodo_f":
        from timm.models import resnetv2

        model_ft = resnetv2.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p7_f":
        from timm.models import resnetv2_p7

        model_ft = resnetv2_p7.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p6_f":
        from timm.models import resnetv2_p6

        model_ft = resnetv2_p6.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p5_f":
        from timm.models import resnetv2_p5

        model_ft = resnetv2_p5.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p4_f":
        from timm.models import resnetv2_p4

        model_ft = resnetv2_p4.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p3_f":
        from timm.models import resnetv2_p3

        model_ft = resnetv2_p3.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p2_f":
        from timm.models import resnetv2_p2

        model_ft = resnetv2_p2.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p1_f":
        from timm.models import resnetv2_p1

        model_ft = resnetv2_p1.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_p0_f":
        from timm.models import resnetv2_p0

        model_ft = resnetv2_p0.resnetv2_50(pretrained=False, num_classes=num_classes)

    ### 101
    if model_name == "my_resnetv2_101_nodo_f":
        from timm.models import resnetv2

        model_ft = resnetv2.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p7_f":
        from timm.models import resnetv2_p7

        model_ft = resnetv2_p7.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p6_f":
        from timm.models import resnetv2_p6

        model_ft = resnetv2_p6.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p5_f":
        from timm.models import resnetv2_p5

        model_ft = resnetv2_p5.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p4_f":
        from timm.models import resnetv2_p4

        model_ft = resnetv2_p4.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p3_f":
        from timm.models import resnetv2_p3

        model_ft = resnetv2_p3.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p2_f":
        from timm.models import resnetv2_p2

        model_ft = resnetv2_p2.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p1_f":
        from timm.models import resnetv2_p1

        model_ft = resnetv2_p1.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_p0_f":
        from timm.models import resnetv2_p0

        model_ft = resnetv2_p0.resnetv2_101(pretrained=False, num_classes=num_classes)

    ### 152
    if model_name == "my_resnetv2_152_nodo_f":
        from timm.models import resnetv2

        model_ft = resnetv2.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p7_f":
        from timm.models import resnetv2_p7

        model_ft = resnetv2_p7.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p6_f":
        from timm.models import resnetv2_p6

        model_ft = resnetv2_p6.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p5_f":
        from timm.models import resnetv2_p5

        model_ft = resnetv2_p5.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p4_f":
        from timm.models import resnetv2_p4

        model_ft = resnetv2_p4.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p3_f":
        from timm.models import resnetv2_p3

        model_ft = resnetv2_p3.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p2_f":
        from timm.models import resnetv2_p2

        model_ft = resnetv2_p2.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p1_f":
        from timm.models import resnetv2_p1

        model_ft = resnetv2_p1.resnetv2_152(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_152_p0_f":
        from timm.models import resnetv2_p0

        model_ft = resnetv2_p0.resnetv2_152(pretrained=False, num_classes=num_classes)

    # mobilenetv2_100
    if model_name == "timm_mobilenetv2_100":
        model_ft = timm.create_model(
            "mobilenetv2_100", pretrained=True, num_classes=num_classes
        )

    if model_name == "timm_mobilenetv2_100_f":
        model_ft = timm.create_model(
            "mobilenetv2_100", pretrained=False, num_classes=num_classes
        )

    # h4, h5, nodo
    if model_name == "my_mobilenetv2_100_h4_f":
        from timm.models import efficientnet_h4

        model_ft = efficientnet_h4.mobilenetv2_100(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_mobilenetv2_100_h5_f":
        from timm.models import efficientnet_h5

        model_ft = efficientnet_h5.mobilenetv2_100(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_mobilenetv2_100_nodo_f":
        from timm.models import efficientnet_nodo

        model_ft = efficientnet_nodo.mobilenetv2_100(
            pretrained=False, num_classes=num_classes
        )

    # h4, h5, nodo
    if model_name == "my_mobilenetv2_140_h4_f":
        from timm.models import efficientnet_h4

        model_ft = efficientnet_h4.mobilenetv2_140(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_mobilenetv2_140_h5_f":
        from timm.models import efficientnet_h5

        model_ft = efficientnet_h5.mobilenetv2_140(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_mobilenetv2_140_nodo_f":
        from timm.models import efficientnet_nodo

        model_ft = efficientnet_nodo.mobilenetv2_140(
            pretrained=False, num_classes=num_classes
        )

    # tf_efficientnetv2_b0
    if model_name == "my_tf_efficientnetv2_b0_h4_f":
        from timm.models import efficientnet_h4

        model_ft = efficientnet_h4.tf_efficientnetv2_b0(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_tf_efficientnetv2_b0_h5_f":
        from timm.models import efficientnet_h5

        model_ft = efficientnet_h5.tf_efficientnetv2_b0(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_tf_efficientnetv2_b0_nodo_f":
        from timm.models import efficientnet_nodo

        model_ft = efficientnet_nodo.tf_efficientnetv2_b0(
            pretrained=False, num_classes=num_classes
        )

    # tf_efficientnet_b0
    if model_name == "my_tf_efficientnet_b0_h4_f":
        from timm.models import efficientnet_h4

        model_ft = efficientnet_h4.tf_efficientnet_b0(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_tf_efficientnet_b0_h5_f":
        from timm.models import efficientnet_h5

        model_ft = efficientnet_h5.tf_efficientnet_b0(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_tf_efficientnet_b0_nodo_f":
        from timm.models import efficientnet_nodo

        model_ft = efficientnet_nodo.tf_efficientnet_b0(
            pretrained=False, num_classes=num_classes
        )

    # efficientnet_b0
    if model_name == "my_efficientnet_b0_h4_f":
        from timm.models import efficientnet_h4

        model_ft = efficientnet_h4.efficientnet_b0(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_efficientnet_b0_h5_f":
        from timm.models import efficientnet_h5

        model_ft = efficientnet_h5.efficientnet_b0(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_efficientnet_b0_nodo_f":
        from timm.models import efficientnet_nodo

        model_ft = efficientnet_nodo.efficientnet_b0(
            pretrained=False, num_classes=num_classes
        )

    # mnasnet_100
    if model_name == "my_mnasnet_100_h4_f":
        from timm.models import efficientnet_h4

        model_ft = efficientnet_h4.mnasnet_100(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_mnasnet_100_h5_f":
        from timm.models import efficientnet_h5

        model_ft = efficientnet_h5.mnasnet_100(
            pretrained=False, num_classes=num_classes
        )

    if model_name == "my_mnasnet_100_nodo_f":
        from timm.models import efficientnet_nodo

        model_ft = efficientnet_nodo.mnasnet_100(
            pretrained=False, num_classes=num_classes
        )

    # resnetrs50
    if model_name == "my_resnetrs50_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnetrs50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetrs50_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnetrs50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetrs50_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnetrs50(pretrained=False, num_classes=num_classes)

    # resnet50
    if model_name == "my_resnet50_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnet50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnet50_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnet50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnet50_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnet50(pretrained=False, num_classes=num_classes)

    # resnet101
    if model_name == "my_resnet101_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnet101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnet101_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnet101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnet101_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnet101(pretrained=False, num_classes=num_classes)

    # resnet50d
    if model_name == "my_resnet50d_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnet50d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnet50d_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnet50d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnet50d_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnet50d(pretrained=False, num_classes=num_classes)

    # wide_resnet101_2
    if model_name == "my_wide_resnet101_2_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.wide_resnet101_2(pretrained=False, num_classes=num_classes)

    if model_name == "my_wide_resnet101_2_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.wide_resnet101_2(pretrained=False, num_classes=num_classes)

    if model_name == "my_wide_resnet101_2_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.wide_resnet101_2(
            pretrained=False, num_classes=num_classes
        )

    # resnext50_32x4d
    if model_name == "my_resnext50_32x4d_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnext50_32x4d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext50_32x4d_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnext50_32x4d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext50_32x4d_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnext50_32x4d(
            pretrained=False, num_classes=num_classes
        )

    # resnext101_32x4d
    if model_name == "my_resnext101_32x4d_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnext101_32x4d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext101_32x4d_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnext101_32x4d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext101_32x4d_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnext101_32x4d(
            pretrained=False, num_classes=num_classes
        )

    # resnext101_32x8d
    if model_name == "my_resnext101_32x8d_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnext101_32x8d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext101_32x8d_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnext101_32x8d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext101_32x8d_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnext101_32x8d(
            pretrained=False, num_classes=num_classes
        )

    # resnext101_64x4d
    if model_name == "my_resnext101_64x4d_h4_f":
        from timm.models import resnet_h4

        model_ft = resnet_h4.resnext101_64x4d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext101_64x4d_h5_f":
        from timm.models import resnet_h5

        model_ft = resnet_h5.resnext101_64x4d(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnext101_64x4d_nodo_f":
        from timm.models import resnet_nodo

        model_ft = resnet_nodo.resnext101_64x4d(
            pretrained=False, num_classes=num_classes
        )

    # densenet121
    if model_name == "my_densenet121_h4_f":
        from timm.models import densenet_h4

        model_ft = densenet_h4.densenet121(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet121_h5_f":
        from timm.models import densenet_h5

        model_ft = densenet_h5.densenet121(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet121_nodo_f":
        from timm.models import densenet  # no nodo

        model_ft = densenet.densenet121(pretrained=False, num_classes=num_classes)

    # densenet169
    if model_name == "my_densenet169_h4_f":
        from timm.models import densenet_h4

        model_ft = densenet_h4.densenet169(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet169_h5_f":
        from timm.models import densenet_h5

        model_ft = densenet_h5.densenet169(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet169_nodo_f":
        from timm.models import densenet  # no nodo

        model_ft = densenet.densenet169(pretrained=False, num_classes=num_classes)

    # densenet201
    if model_name == "my_densenet201_h4_f":
        from timm.models import densenet_h4

        model_ft = densenet_h4.densenet201(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet201_h5_f":
        from timm.models import densenet_h5

        model_ft = densenet_h5.densenet201(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet201_nodo_f":
        from timm.models import densenet  # no nodo

        model_ft = densenet.densenet201(pretrained=False, num_classes=num_classes)

    # densenet161
    if model_name == "my_densenet161_h4_f":
        from timm.models import densenet_h4

        model_ft = densenet_h4.densenet161(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet161_h5_f":
        from timm.models import densenet_h5

        model_ft = densenet_h5.densenet161(pretrained=False, num_classes=num_classes)

    if model_name == "my_densenet161_nodo_f":
        from timm.models import densenet  # no nodo

        model_ft = densenet.densenet161(pretrained=False, num_classes=num_classes)

    # regnetx_320
    if model_name == "my_regnetx_320_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_320(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_320_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_320(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_320_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_320(pretrained=False, num_classes=num_classes)

    # regnetx_160
    if model_name == "my_regnetx_160_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_160(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_160_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_160(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_160_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_160(pretrained=False, num_classes=num_classes)

    # regnetx_120
    if model_name == "my_regnetx_120_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_120(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_120_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_120(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_120_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_120(pretrained=False, num_classes=num_classes)

    # regnetx_080
    if model_name == "my_regnetx_080_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_080(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_080_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_080(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_080_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_080(pretrained=False, num_classes=num_classes)

    # regnetx_064
    if model_name == "my_regnetx_064_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_064(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_064_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_064(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_064_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_064(pretrained=False, num_classes=num_classes)

    # regnetx_040
    if model_name == "my_regnetx_040_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_040(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_040_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_040(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_040_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_040(pretrained=False, num_classes=num_classes)

    # regnetx_032
    if model_name == "my_regnetx_032_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_032(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_032_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_032(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_032_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_032(pretrained=False, num_classes=num_classes)

    # regnetx_016
    if model_name == "my_regnetx_016_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_016(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_016_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_016(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_016_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_016(pretrained=False, num_classes=num_classes)

    # regnetx_008
    if model_name == "my_regnetx_008_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_008(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_008_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_008(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_008_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_008(pretrained=False, num_classes=num_classes)

    # regnetx_006
    if model_name == "my_regnetx_006_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_006(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_006_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_006(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_006_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_006(pretrained=False, num_classes=num_classes)

    # regnetx_004
    if model_name == "my_regnetx_004_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_004(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_004_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_004(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_004_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_004(pretrained=False, num_classes=num_classes)

    # regnetx_002
    if model_name == "my_regnetx_002_h4_f":
        from timm.models import regnet_h4

        model_ft = regnet_h4.regnetx_002(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_002_h5_f":
        from timm.models import regnet_h5

        model_ft = regnet_h5.regnetx_002(pretrained=False, num_classes=num_classes)

    if model_name == "my_regnetx_002_nodo_f":
        from timm.models import regnet_nodo

        model_ft = regnet_nodo.regnetx_002(pretrained=False, num_classes=num_classes)

    # resnetv2_50
    if model_name == "my_resnetv2_50_h4_f":
        from timm.models import resnetv2_h4

        model_ft = resnetv2_h4.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_h5_f":
        from timm.models import resnetv2_h5

        model_ft = resnetv2_h5.resnetv2_50(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_50_nodo2_f":
        from timm.models import resnetv2_nodo

        model_ft = resnetv2_nodo.resnetv2_50(pretrained=False, num_classes=num_classes)

    # resnetv2_101
    if model_name == "my_resnetv2_101_h4_f":
        from timm.models import resnetv2_h4

        model_ft = resnetv2_h4.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_h5_f":
        from timm.models import resnetv2_h5

        model_ft = resnetv2_h5.resnetv2_101(pretrained=False, num_classes=num_classes)

    if model_name == "my_resnetv2_101_nodo2_f":
        from timm.models import resnetv2_nodo

        model_ft = resnetv2_nodo.resnetv2_101(pretrained=False, num_classes=num_classes)

    # last fc configuration
    if model_type == "res":
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
    elif model_type == "vgg":
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
    elif model_type == "den":
        num_ftrs = model_ft.classifier.in_features
        model_ft._fc = nn.Linear(num_ftrs, num_classes)

    return model_ft
