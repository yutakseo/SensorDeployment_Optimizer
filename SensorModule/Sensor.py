import torch


class Sensor:
    FIXED_STRENGTH = 10.0

    def __init__(self, MAP, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(MAP, torch.Tensor):
            map_tensor = MAP.to(device=self.device, dtype=torch.float32)
        else:
            map_tensor = torch.tensor(MAP, device=self.device, dtype=torch.float32)

        self.map_tensor = map_tensor
        self.MAP = map_tensor.clone()

        self.radius = None
        self.circle = None
        self._offsets = None

        self.sensor_log = []

    def _create_circle(self, radius: int) -> torch.Tensor:
        d = 2 * radius + 1
        y, x = torch.meshgrid(
            torch.arange(d, device=self.device),
            torch.arange(d, device=self.device),
            indexing="ij",
        )
        c = radius
        dist = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
        return (dist <= radius).float()

    def _ensure_kernel(self, coverage: int) -> None:
        if (coverage != self.radius) or (self.circle is None):
            self.circle = self._create_circle(coverage)
            self.radius = coverage
            ys, xs = torch.where(self.circle > 0)
            self._offsets = torch.stack([ys - coverage, xs - coverage], dim=1).to(torch.long)

    @torch.no_grad()
    def reset(self):
        self.MAP.copy_(self.map_tensor)
        self.sensor_log.clear()
        return self.MAP

    @torch.no_grad()
    def deploy(self, sensor_position, coverage: int = 45):
        cov = int(coverage / 5)
        self._ensure_kernel(cov)

        H, W = self.MAP.shape
        strength = float(self.FIXED_STRENGTH)

        if isinstance(sensor_position, tuple) and len(sensor_position) == 2:
            pos = torch.tensor([sensor_position], device=self.device, dtype=torch.long)
        else:
            pos = torch.tensor(sensor_position, device=self.device, dtype=torch.long)

        for x, y in pos.tolist():
            self.sensor_log.append((x, y, cov))

        x0 = pos[:, 0]
        y0 = pos[:, 1]
        dy = self._offsets[:, 0]
        dx = self._offsets[:, 1]

        yy = y0[:, None] + dy[None, :]
        xx = x0[:, None] + dx[None, :]

        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)

        lin = (yy * W + xx)[valid]
        values = torch.full((lin.numel(),), strength, device=self.device, dtype=self.MAP.dtype)

        self.MAP.view(-1).scatter_add_(0, lin, values)
        return self.MAP

    def remove(self, sensor_position):
        if not (isinstance(sensor_position, tuple) and len(sensor_position) == 2):
            out = self.MAP
            for p in sensor_position:
                out = self.remove(tuple(p))
            return out

        x, y = sensor_position

        idx = None
        cov = None
        for i in range(len(self.sensor_log) - 1, -1, -1):
            sx, sy, scov = self.sensor_log[i]
            if sx == x and sy == y:
                idx = i
                cov = scov
                break

        if idx is None:
            return self.MAP

        self.sensor_log.pop(idx)

        self._ensure_kernel(cov)

        H, W = self.MAP.shape
        strength = -float(self.FIXED_STRENGTH)

        pos = torch.tensor([(x, y)], device=self.device, dtype=torch.long)
        x0 = pos[:, 0]
        y0 = pos[:, 1]
        dy = self._offsets[:, 0]
        dx = self._offsets[:, 1]

        yy = y0[:, None] + dy[None, :]
        xx = x0[:, None] + dx[None, :]

        valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)

        lin = (yy * W + xx)[valid]
        values = torch.full((lin.numel(),), strength, device=self.device, dtype=self.MAP.dtype)

        self.MAP.view(-1).scatter_add_(0, lin, values)
        self.MAP = torch.clamp(self.MAP, min=0.0)
        return self.MAP

    def extract_only_sensor(self):
        mask = (self.MAP > self.FIXED_STRENGTH).float()
        return self.MAP * mask
