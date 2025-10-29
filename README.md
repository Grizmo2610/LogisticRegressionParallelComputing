# Logistic Regression Implementation (C++ / OpenMP / Python)

## Version Requirements
- CMake 3.10.2
- Python 3.10.2
- C++17 compiler with OpenMP support (g++)
- make (Unix Makefiles)
- pybind11 (for Python-C++ interface)

## Project Structure
```
.
├── CMakeLists.txt
├── include/
│   ├── model.h
│   └── utils.h
├── src/
│   ├── main.cpp
│   ├── model.cpp
│   └── utils.cpp
└── pythoncode/
    ├── model.py
    ├── requirements.txt
    └── test.py
```

## Build Instructions

### Important CMake Configuration
Before building, you need to modify `CMakeLists.txt`:

1. For C++ executable, ensure these lines are configured:
```cmake
# Comment out the Python module build
# pybind11_add_module(Logistic ${SOURCES})

# Uncomment the executable build
add_executable(Logistic ${SOURCES})
```

2. For Python module, ensure these lines are configured:
```cmake
# Comment out the executable build
# add_executable(Logistic ${SOURCES})

# Uncomment the Python module build
pybind11_add_module(Logistic ${SOURCES})
```

### Option 1: Standalone C++ Executable

1. Create and enter build directory:
```bash
mkdir -p build
cd build
```

2. Configure CMake:
```bash
cmake ..
```

3. Build the project:
```bash
make
```

### Option 2: Python Module Build

1. Install Python dependencies:
```bash
python3.10 -m pip install -r pythoncode/requirements.txt
python3.10 -m pip install pybind11
```

2. Build as described in Option 1

## Running the Code

### C++ Executable
From the build directory:

```bash
# Basic usage
./Logistic

# With specific parameters
./Logistic --dataset <path_to_dataset> \
           --epochs <number_of_epochs> \
           --learning_rate <learning_rate> \
           --batch_size <batch_size> \
           --threads <num_threads>

# Example with common parameters
./Logistic --dataset "../data/100k_1k/train.csv" \
           --epochs 100 \
           --learning_rate 0.01 \
           --batch_size 32 \
           --threads 4
```

Available parameters:
- `--dataset`: Path to training data (CSV format)
- `--epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate for gradient descent (default: 0.01)
- `--batch_size`: Mini-batch size (default: 32)
- `--threads`: Number of OpenMP threads (default: 4)

### Python Integration
From the project root:
```bash
python3.10 pythoncode/test.py
```

## Performance Notes
- OpenMP parallelization is enabled with -O2 optimization
- Performance scales with the number of threads specified
- Recommended to set threads according to available CPU cores

## Platform Support
- C++ implementation: Linux only
- Python integration: Linux only
- Windows support is currently under development

## Troubleshooting
1. If CMake can't find Python:
```bash
export Python_ROOT_DIR=/usr/local/python3.10.2
cmake ..
```

2. If OpenMP is not found:
```bash
sudo apt-get install libomp-dev  # For Ubuntu/Debian
```