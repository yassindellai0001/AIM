cmake_minimum_required(VERSION 3.10)
project(PC_Aimbot)

# إجبار المشروع على البناء كـ Console Application وليس GUI
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# البحث عن OpenCV
find_package(OpenCV REQUIRED)

# إضافة ملف الكود المصدري
add_executable(PC_Aimbot main.cpp)

# تضمين مسارات OpenCV
target_include_directories(PC_Aimbot PRIVATE ${OpenCV_INCLUDE_DIRS})

# ربط مكتبات OpenCV و DirectX
target_link_libraries(PC_Aimbot PRIVATE 
    ${OpenCV_LIBS} 
    d3d11 
    dxgi 
    gdi32 
    user32
)
