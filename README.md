# Dự án Huấn Luyện Agent Sử Dụng Thuật Toán QMIX

![Project Image](https://example.com/project-image.png)

## Giới thiệu

Dự án này nhằm mục đích huấn luyện các agent sử dụng thuật toán QMIX trong môi trường chiến đấu của MAgent2. PyTorch được sử dụng để xây dựng và huấn luyện các mô hình học sâu, với mục tiêu tối ưu hóa hiệu suất và tăng cường khả năng phối hợp giữa các agent.

## Tính năng

- Sử dụng thuật toán QMIX để huấn luyện các agent.
- Môi trường chiến đấu được mô phỏng bằng MAgent2.
- Tối ưu hóa quá trình huấn luyện bằng GPU.
- Lưu trữ và phân tích kết quả huấn luyện dưới dạng video.

## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- MAgent2

### Hướng dẫn cài đặt

1. Clone repository:

    ```bash
    git clone https://github.com/username/project-name.git
    cd project-name
    ```

2. Cài đặt các thư viện cần thiết:

    ```bash
    pip install -r requirements.txt
    ```

## Sử dụng

### Khởi động

1. Chạy tệp `main.py` để bắt đầu quá trình huấn luyện:

    ```bash
    python main.py
    ```

2. Kết quả huấn luyện sẽ được lưu trong thư mục `video`.

### Cấu trúc dự án

- `main.py`: Tệp điều khiển chính, khởi tạo môi trường và huấn luyện các agent.
- `torch_model.py`: Định nghĩa các lớp mạng neural và Replay Buffer.
- `requirements.txt`: Danh sách các thư viện cần thiết cho dự án.

### Ví dụ

Dưới đây là một ví dụ về cách sử dụng dự án này để huấn luyện một tập hợp các agent:

```python
from magent2.environments import battle_v4
from torch_model import QNetwork, QMixer, ReplayBuffer
import torch.optim as optim
import numpy as np

# Khởi tạo môi trường
env = battle_v4.env(map_size=45, render_mode="rgb_array")

# Thiết lập các tham số huấn luyện
num_episodes = 30
max_steps = 300
batch_size = 32
gamma = 0.99

# Khởi tạo mạng neural và Replay Buffer
n_agents = 81
observation_shape = env.observation_space("blue_0").shape
action_space = env.action_space("blue_0").n
state_shape = env.state_space.shape
state_dim = np.prod(state_shape)

agent_networks = [QNetwork(observation_shape, action_space).to(device) for _ in range(n_agents)]
qmixer = QMixer(n_agents, state_dim).to(device)
replay_buffer = ReplayBuffer(capacity=10000, observation_shape=observation_shape)
agent_optimizers = [optim.Adam(network.parameters(), lr=0.001) for network in agent_networks]
qmixer_optimizer = optim.Adam(qmixer.parameters(), lr=0.001)

# Huấn luyện các agent (chi tiết trong main.py)

Đóng góp
Chúng tôi hoan nghênh mọi đóng góp cho dự án này. Nếu bạn có ý tưởng, báo lỗi hoặc muốn tham gia phát triển, hãy mở một issue hoặc pull request trên GitHub.

### Giải thích các phần chính của README.md

1. **Tiêu đề và Hình ảnh Dự án:**
   - Tiêu đề rõ ràng và hình ảnh minh họa giúp người đọc hiểu nhanh về dự án.

2. **Giới thiệu:**
   - Mô tả ngắn gọn về mục tiêu và phạm vi của dự án.

3. **Tính năng:**
   - Liệt kê các tính năng chính của dự án.

4. **Cài đặt:**
   - Yêu cầu hệ thống và hướng dẫn cài đặt chi tiết.

5. **Sử dụng:**
   - Hướng dẫn cách khởi động và sử dụng dự án, cùng với một ví dụ cụ thể.

6. **Cấu trúc dự án:**
   - Giải thích ngắn gọn về các tệp và thư mục chính trong dự án.

7. **Đóng góp:**
   - Hướng dẫn cách đóng góp vào dự án.

8. **Giấy phép:**
   - Thông tin về giấy phép sử dụng của dự án.

9. **Liên hệ:**
   - Cung cấp thông tin liên hệ để người dùng có thể liên hệ khi cần thiết.

Hy vọng mẫu README.md này sẽ giúp bạn tạo ra một tài liệu hướng dẫn chi tiết và chuyên nghiệp cho dự án của mình! Nếu bạn cần thêm thông tin hoặc hỗ trợ, đừng ngần ngại hỏi.
