# HƯỚNG DẪN CHẠY INFERENCE TRÊN PYTHON

## 1. Hệ điều hành

### Linux (Khuyến nghị)

Model C++ được tối ưu cho Linux. Nên chạy trên Linux hoặc môi trường tương thích như WSL2 trên Windows.

### Windows (Tùy chọn)

Có thể build trực tiếp sang `.pyd` bằng MinGW-w64.
Tải MinGW-w64 từ [https://winlibs.com/#download-release](https://winlibs.com/#download-release).

Tìm mục **MSVCRT runtime**, kéo xuống phía dưới và tải phiên bản mới nhất hoặc phiên bản [15.2](https://github.com/brechtsanders/winlibs_mingw/releases/download/15.2.0posix-13.0.0-msvcrt-r2/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r2.zip).

Giải nén vào `C:\mingw64` và thêm thư mục `bin` vào `PATH` (ví dụ: `C:\mingw64\bin`).

Cấu hình khi cài:

```
Architecture: x86_64
Threads: posix
Exception: seh
```

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

File tạo ra: `Logistic.cpython-xx-x86_64-linux-gnu.so`  *(xx thay thế phiên bản Python hiện tại, ví dụ 311 cho Python 3.11)*
Copy vào cùng thư mục Python code.

### Windows (MinGW)

```bash
mkdir build
cd build

# Chạy CMake trên Windows với MinGW và đường dẫn tới pybind11
cmake -G "MinGW Makefiles" -Dpybind11_DIR="C:/Users/Hoang Tu/AppData/Local/Programs/Python/Pythonxx/Lib/site-packages/pybind11/share/cmake/pybind11" ..

# Biên dịch module
cmake --build .
```

* Thay `Hoang Tu` bằng tên user Windows của bạn.
* Thay `Pythonxx` bằng phiên bản Python đang dùng (xx).

Kết quả: `Logistic.cp**xx**-win_amd64.pyd`
Copy vào cùng thư mục Python code để import.

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

## 6. Sử dụng trực tiếp model mà không cần biên dịch

Nếu đã có sẵn file `.so` (Linux) hoặc `.pyd` (Windows), có thể bỏ qua bước biên dịch và dùng trực tiếp thông qua một file `model.py` khởi tạo sẵn model.

**Lưu ý:** Trên Windows, việc cài đặt **MinGW-w64** là bắt buộc để có thể chạy module `.pyd` đúng cách (xem **phần 1 – Windows**).

**model.py**

```python
import os
import platform

if platform.system().lower() == "windows":
    os.add_dll_directory(r"C:\mingw64\bin")  # Tham khảo phần 1 – Windows

import Logistic

# Khởi tạo model
model = Logistic.LogisticRegression(core)
```

Sau đó chỉ cần import `model` trong các file Python khác:

```python
from model import model

# Sử dụng model
predictions = model.predict(data)
```