import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DualStreamUNet_MS(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()

        # 1. RGB branch
        self.rgb_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=1,
            in_channels=3,
        )

        # 2. gray branch
        self.gray_encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=1,
            depth=5,
            weights=encoder_weights
        )

        full_channels = self.rgb_model.encoder.out_channels

        channels = full_channels[1:]

        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in channels
        ])

    def forward(self, x_rgb, x_gray):
        # 1. Feature Flow Extraction
        rgb_feats = self.rgb_model.encoder(x_rgb)
        gray_feats = self.gray_encoder(x_gray)

        # 2. Layer-by-layer multiscale fusion
        fused_feats = []
        for i in range(len(rgb_feats)):
            if i == 0:
                fused_feats.append(rgb_feats[i])
                continue

            # concat
            combined = torch.cat([rgb_feats[i], gray_feats[i]], dim=1)

            fused = self.fusion_layers[i - 1](combined)
            fused_feats.append(fused)

        decoder_output = self.rgb_model.decoder(fused_feats)

        output = self.rgb_model.segmentation_head(decoder_output)

        return output


# --- 1. CBAM  ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        res = self.conv1(res)
        return self.sigmoid(res)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# --- 2. DualStreamUNet_CBAM ---
class DualStreamUNet_CBAM(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()

        self.rgb_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=1,
            in_channels=3,
        )

        self.gray_encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=1, weights=encoder_weights
        )

        full_channels = self.rgb_model.encoder.out_channels

        fusion_channels = full_channels[1:]

        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, kernel_size=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in fusion_channels
        ])

        self.attention_block = CBAM(fusion_channels[-1])

    def forward(self, x_rgb, x_gray):
        rgb_feats = self.rgb_model.encoder(x_rgb)
        gray_feats = self.gray_encoder(x_gray)

        fused_feats = []
        for i in range(len(rgb_feats)):
            if i == 0:
                fused_feats.append(rgb_feats[i])
                continue

            combined = torch.cat([rgb_feats[i], gray_feats[i]], dim=1)

            fused = self.fusion_layers[i - 1](combined)

            if i == len(rgb_feats) - 1:
                fused = self.attention_block(fused)

            fused_feats.append(fused)

        decoder_output = self.rgb_model.decoder(fused_feats)
        output = self.rgb_model.segmentation_head(decoder_output)

        return output

# ---  CoordAtt  ---
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


# --- DualStreamUNet_CA ---
class DualStreamUNet_CA(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet"):
        super().__init__()
        self.rgb_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=1,
            in_channels=3,
        )

        self.gray_encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=1, weights="imagenet"
        )

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CoordAtt(512, 512)
        )

    def forward(self, x_rgb, x_gray):
        rgb_features = self.rgb_model.encoder(x_rgb)
        gray_features = self.gray_encoder(x_gray)

        combined = torch.cat([rgb_features[-1], gray_features[-1]], dim=1)

        fused = self.fusion_layer(combined)

        rgb_features_list = list(rgb_features)
        rgb_features_list[-1] = fused

        decoder_output = self.rgb_model.decoder(rgb_features_list)
        output = self.rgb_model.segmentation_head(decoder_output)

        return output