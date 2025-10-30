# HƯỚNG DẪN CHẠY INFERENCE TRÊN PYTHON

## 1. Hệ điều hành

### Linux (Khuyến nghị)

Model C++ được tối ưu cho Linux. Nên chạy trên Linux hoặc môi trường tương thích như WSL2 trên Windows.

### Windows (Tùy chọn)

Có thể build trực tiếp sang `.pyd` bằng MinGW-w64.
Tải MinGW-w64 tại [https://www.mingw-w64.org/downloads/](https://www.mingw-w64.org/downloads/)
Cấu hình khi cài:

```
Architecture: x86_64
Threads: posix
Exception: seh
```

Thêm `C:\mingw64\bin` vào `PATH`.

## 2. Cấu hình Python

Cài các thư viện:

```bash
pip install -r requirements.txt
```

Các thư viện quan trọng:

* `pybind11` – kết nối C++ với Python
* `cmake` – build dự án
* `setuptools` – cần khi build module lại

## 3. Biên dịch module

### Linux

```bash
mkdir build
cd build
cmake ..
make
```

File tạo ra: `Logistic.cpython-310-x86_64-linux-gnu.so`
Copy vào cùng thư mục Python code.

### Windows (MinGW)

```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
cmake --build .
```

File tạo ra: `Logistic.cp311-win_amd64.pyd`
Copy vào cùng thư mục Python code.

## 4. Import module trong Python

### Linux

```python
import Logistic
model = Logistic.LogisticRegression(core)
```

### Windows

```python
import os, platform

if platform.system().lower() == "windows":
    os.add_dll_directory(r"C:\mingw64\bin")

import Logistic
model = Logistic.LogisticRegression(core)
```

## 5. Sử dụng

Nếu không muốn build, dùng sẵn file trong `pythoncode/`:

```python
import Logistic
model = Logistic.LogisticRegression(core)
```

Module Python sử dụng trực tiếp, lõi xử lý bằng C++ cho hiệu năng cao.
