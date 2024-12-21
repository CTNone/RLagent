# Dự án Huấn Luyện Agent Sử Dụng Thuật Toán QMIX

![Video thumbnail](https://raw.githubusercontent.com/CTNone/RLagent/main/episode_0%20(1).mp4)

# Chu Thân Nhất ( 22022578 )
## Giới thiệu

Dự án này nhằm mục đích huấn luyện các agent sử dụng thuật toán QMIX trong môi trường chiến đấu của MAgent2. PyTorch được sử dụng để xây dựng và huấn luyện các mô hình học sâu, với mục tiêu tối ưu hóa hiệu suất và tăng cường khả năng phối hợp giữa các agent.

## Tính năng

- Sử dụng thuật toán QMIX để huấn luyện các agent.
- Môi trường chiến đấu được mô phỏng bằng MAgent2.
- Tối ưu hóa quá trình huấn luyện bằng GPU.
- Lưu trữ và phân tích kết quả huấn luyện dưới dạng video.


### Hướng dẫn cài đặt

1. Tải file Untitled0.ipynb về và chạy trên colab.
2. Chạy cell 1
   
 %%capture
 
!git clone https://github.com/giangbang/RL-final-project-AIT-3007

%cd RL-final-project-AIT-3007

!pip install -r requirements.txt

3. Chạy cell thứ 2
   
!python main.py để thực hiện train

4. Đánh giá
   
!python eval.py để thực hiện đánh giá 

## Sử dụng

### Khởi động

1. Chạy tệp `main.py` để bắt đầu quá trình huấn luyện:

    ```bash
    python main.py
    ```

2. Kết quả huấn luyện sẽ được lưu trong thư mục `video`.

### Cấu trúc dự án
Bài code được triển khai theo bài hướng dẫn gốc này:

(https://github.com/giangbang/RL-final-project-AIT-3007)

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


Hy vọng mẫu README.md này sẽ giúp bạn tạo ra một tài liệu hướng dẫn chi tiết và chuyên nghiệp cho dự án của mình! Nếu bạn cần thêm thông tin hoặc hỗ trợ, đừng ngần ngại hỏi.
