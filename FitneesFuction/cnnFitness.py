import torch
import torch.nn as nn
import numpy as np


from bot import MAP


class CNNFitness(nn.Module):
    def __init__(self, input_map_tensor):
        super(CNNFitness, self).__init__()
        self.input_map = torch.tensor(input_map_tensor).float()
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv7 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.conv9 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=9, stride=1, padding=4)

        with torch.no_grad():
            self.conv3.weight.fill_(1 / (3 * 3))
            self.conv5.weight.fill_(1 / (5 * 5))
            self.conv7.weight.fill_(1 / (7 * 7))
            self.conv9.weight.fill_(1 / (9 * 9))
            if self.conv3.bias is not None:
                self.conv3.bias.zero_()
            if self.conv5.bias is not None:
                self.conv5.bias.zero_()
            if self.conv7.bias is not None:
                self.conv7.bias.zero_()
            if self.conv9.bias is not None:
                self.conv9.bias.zero_()

    def forward(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        out9 = self.conv9(x)
        out = (out3 + out5 + out7 + out9) / 4.0
        return out * self.input_map  # 입력맵과 element-wise 곱

    
    def fitness(self, corner_sensors, gene_sensors, coverage):
        return None
    
    
temp = 0

def show_output(output_tensor, title="After Conv2D"):
    import matplotlib.pyplot as plt
    global temp
    plt.figure(figsize=(24, 16), dpi=600)
    # 텐서인 경우 numpy로 변환
    if isinstance(output_tensor, torch.Tensor):
        output_tensor = output_tensor.squeeze().numpy()
    plt.imshow(output_tensor, cmap='jet')
    plt.title(title)
    plt.colorbar(label='Activation')
    plt.tight_layout()
    temp += 1
    filename = f"__RESULTS__/output_{temp}.png"
    plt.savefig(filename)
    print(f"✅ 저장됨: {filename}")
    plt.close()  # 메모리 누수 방지!
    

# 맵데이터 입력   
np_map = np.array(MAP)
tensor_map = torch.from_numpy(np_map).unsqueeze(0).unsqueeze(0).float()
print(tensor_map.shape)  # ✅ torch.Size([1, 1, 10, 16])


# 3. Conv 연산 수행
model = CNNFitness(MAP)
with torch.no_grad():
    output_tensor = model(tensor_map)
    
    
# 4. 시각화
show_output(output_tensor)
show_output(MAP)
