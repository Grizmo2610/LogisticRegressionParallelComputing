# Logistic (C++ / OpenMP)

Miêu tả ngắn
- Dự án này là một chương trình C++ biên dịch thành thực thi tên `main`. Dự án dùng CMake (>= 3.10) và bật tối ưu + OpenMP theo mặc định.
- Tài liệu tham khảo / nghiên cứu liên quan được tập trung trong [doc.md](doc.md).

Yêu cầu
- CMake >= 3.10
- Trình biên dịch C++17 có hỗ trợ OpenMP (ví dụ g++ trên Linux)
- make (Unix Makefiles)

Cấu trúc dự án
- [CMakeLists.txt](CMakeLists.txt) — cấu hình CMake; bật C++17 và thêm flag `-O2 -fopenmp`.
- [documents.md](documents.md) — link tới các tài liệu tham khảo/giải thích (tài liệu ngoại vi).
- include/
  - [include/model.h](include/model.h)
  - [include/utils.h](include/utils.h)
- src/
  - [src/main.cpp](src/main.cpp)
  - [src/model.cpp](src/model.cpp)
  - [src/utils.cpp](src/utils.cpp)
- build/ — thư mục build được CMake tạo, chứa:
  - [build/Makefile](build/Makefile)
  - [build/CMakeFiles/main.dir/build.make](build/CMakeFiles/main.dir/build.make)
  - [build/CMakeFiles/main.dir/DependInfo.cmake](build/CMakeFiles/main.dir/DependInfo.cmake)
- VSCode config: [.vscode/settings.json](.vscode/settings.json)

Hướng dẫn build & chạy
1. Tạo thư mục build và chạy CMake:
   ```sh
   mkdir -p build
   cd build
   cmake ..
   ```