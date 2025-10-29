# **HƯỚNG DẪN CHẠY INFERENCE TREEN PYTHON**

## Hệ điều hành

Do model được xây dựng về phía C++ được viết trên Linux, nên Hệ điều hành yêu cầu sử dụng bắt buộc phải là **linux**. Trong trường hợp máy người dùng là hệ điều window cần cài WSL (Windows Subsystem for Linux) để sử dụng

## Cấu hình
Cài đặt các thư viện cần thiết thông qua file `requirements.txt`. Trong đó quan trọng nhất là `pybind11` để đảm bảo việc code qua python

```bash
pip install -r requirements.txt
```

## Biên dịch

Đối với việc sửa đổi code C++ hoặc chưa có file `.so` cần thực hiện như sau:

Tại thư mục gốc thực hiện các câu lệnh sau trene linux

```bash
mkdir build
cd build
cmake ..
make
```

Khi này trong thư mục build sẽ có 1 file: `Logistic.cpython-310-x86_64-linux-gnu.so`. Copy hoặc di chuyển file này sang thư mục chứa python code để import thư viện không bị lỗi.

Sau đó tại python code
```python

import Logistic
model = Logistic.LogisticRegression(core)
```

hoặc đơn giản hơn thì sử dụng trực tiếp các file đã có sẵn tại thư mục `pythoncode` đã có đầy đủ.