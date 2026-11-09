from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent

INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
    ):
        super().__init__()

        self.register_buffer(
            "input_mean",
            torch.as_tensor(INPUT_MEAN)
        )

        self.register_buffer(
            "input_std",
            torch.as_tensor(INPUT_STD)
        )

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = (
            x - self.input_mean[None, :, None, None]
        ) / self.input_std[None, :, None, None]

        z = self.features(z)

        logits = self.classifier(z)

        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(dim=1)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                out_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):

        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat([x, skip], dim=1)

        return self.conv(x)


class Detector(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
    ):
        super().__init__()

        self.register_buffer(
            "input_mean",
            torch.as_tensor(INPUT_MEAN)
        )

        self.register_buffer(
            "input_std",
            torch.as_tensor(INPUT_STD)
        )

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)

        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up3 = UpBlock(64, 32, 32)

        self.seg_head = nn.Conv2d(
            32,
            num_classes,
            kernel_size=1,
        )

        self.depth_head = nn.Conv2d(
            32,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        z = (
            x - self.input_mean[None, :, None, None]
        ) / self.input_std[None, :, None, None]

        s0 = self.stem(z)

        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)

        u1 = self.up1(s3, s2)
        u2 = self.up2(u1, s1)
        u3 = self.up3(u2, s0)

        logits = self.seg_head(u3)

        depth = torch.sigmoid(
            self.depth_head(u3)
        ).squeeze(1)

        return logits, depth

    def predict(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        logits, raw_depth = self(x)

        pred = logits.argmax(dim=1)

        depth = raw_depth

        return pred, depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}

def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """

    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"

        assert model_path.exists(), (
            f"{model_path.name} not found"
        )

        try:
            m.load_state_dict(
                torch.load(
                    model_path,
                    map_location="cpu"
                )
            )
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, "
                "make sure the default model arguments "
                "are set correctly"
            ) from e

    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(
            f"{model_name} is too large: "
            f"{model_size_mb:.2f} MB"
        )

    return m

def save_model(model: torch.nn.Module) -> str:

    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(
            f"Model type '{str(type(model))}' not supported"
        )

    output_path = (
        HOMEWORK_DIR /
        f"{model_name}.th"
    )

    torch.save(
        model.state_dict(),
        output_path,
    )

    return output_path


def calculate_model_size_mb(
    model: torch.nn.Module,
) -> float:

    return (
        sum(
            p.numel()
            for p in model.parameters()
        )
        * 4
        / 1024
        / 1024
    )


def debug_model(
    batch_size: int = 2,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    clf = Classifier().to(device)

    x = torch.rand(
        batch_size,
        3,
        64,
        64
    ).to(device)

    y = clf(x)

    print(
        "Classifier output:",
        y.shape
    )

    det = Detector().to(device)

    x = torch.rand(
        batch_size,
        3,
        96,
        128
    ).to(device)

    logits, depth = det(x)

    print(
        "Segmentation:",
        logits.shape
    )

    print(
        "Depth:",
        depth.shape
    )


if __name__ == "__main__":
    debug_model()