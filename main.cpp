#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <random>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdi32.lib")

// ================= المتغيرات العامة =================
int g_circleRadius = 40; // نصف قطر الدائرة (قابل للتغيير من الكيبورد)
const int TRIGGER_KEY = VK_MENU; // ALT
// =====================================================

// =====================================================================
// محرك الالتقاط (DXGI)
// =====================================================================
class DXGICaptureEngine {
private:
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGIOutputDuplication* deskDupl = nullptr;
    ID3D11Texture2D* stagingTex = nullptr;
    int texWidth = 0;
    int texHeight = 0;

public:
    bool Init() {
        D3D_FEATURE_LEVEL featureLevel;
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &featureLevel, &context);
        if (FAILED(hr)) return false;

        IDXGIDevice* dxgiDevice = nullptr; device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
        IDXGIAdapter* dxgiAdapter = nullptr; dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgiAdapter);
        IDXGIOutput* dxgiOutput = nullptr; dxgiAdapter->EnumOutputs(0, &dxgiOutput);
        IDXGIOutput1* dxgiOutput1 = nullptr; dxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgiOutput1);

        hr = dxgiOutput1->DuplicateOutput(device, &deskDupl);
        dxgiOutput1->Release(); dxgiOutput->Release(); dxgiAdapter->Release(); dxgiDevice->Release();
        return SUCCEEDED(hr);
    }

    int GrabFrame(cv::Mat& outFrame) {
        if (!deskDupl) return -1;
        DXGI_OUTDUPL_FRAME_INFO frameInfo; IDXGIResource* desktopResource = nullptr;
        HRESULT hr = deskDupl->AcquireNextFrame(0, &frameInfo, &desktopResource);
        
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return 0; 
        if (hr == DXGI_ERROR_ACCESS_LOST) return -1; 
        if (FAILED(hr)) return -1;

        ID3D11Texture2D* acquiredDesktopImage = nullptr;
        desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&acquiredDesktopImage);
        D3D11_TEXTURE2D_DESC desc; acquiredDesktopImage->GetDesc(&desc);

        if (!stagingTex || texWidth != desc.Width || texHeight != desc.Height) {
            if (stagingTex) stagingTex->Release();
            desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.BindFlags = 0; desc.MiscFlags = 0;
            device->CreateTexture2D(&desc, nullptr, &stagingTex);
            texWidth = desc.Width; texHeight = desc.Height;
        }

        context->CopyResource(stagingTex, acquiredDesktopImage);
        D3D11_MAPPED_SUBRESOURCE mapped;
        hr = context->Map(stagingTex, 0, D3D11_MAP_READ, 0, &mapped);
        
        if (SUCCEEDED(hr)) {
            cv::Mat fullFrame(desc.Height, desc.Width, CV_8UC4, mapped.pData, mapped.RowPitch);
            int centerX = desc.Width / 2; int centerY = desc.Height / 2;
            
            // نأخذ مربعاً أكبر قليلاً من الدائرة لضمان قراءة حوافها
            int FOV = g_circleRadius + 10; 
            cv::Rect roi(centerX - FOV, centerY - FOV, FOV * 2, FOV * 2);
            
            if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= desc.Width && roi.y + roi.height <= desc.Height) {
                outFrame = fullFrame(roi).clone();
            }
            context->Unmap(stagingTex, 0);
        }
        acquiredDesktopImage->Release(); deskDupl->ReleaseFrame();
        return 1;
    }

    void Release() {
        if (stagingTex) stagingTex->Release(); if (deskDupl) deskDupl->Release();
        if (context) context->Release(); if (device) device->Release();
    }
};

// =====================================================================
// نافذة الرسم (الدائرة الخضراء الشفافة)
// =====================================================================
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        
        // إعداد القلم الأخضر
        HPEN hPen = CreatePen(PS_SOLID, 2, RGB(0, 255, 0));
        HGDIOBJ oldPen = SelectObject(hdc, hPen);
        HBRUSH hBrush = (HBRUSH)GetStockObject(NULL_BRUSH); // شفاف من الداخل
        HGDIOBJ oldBrush = SelectObject(hdc, hBrush);

        int screenW = GetSystemMetrics(SM_CXSCREEN);
        int screenH = GetSystemMetrics(SM_CYSCREEN);
        int cx = screenW / 2;
        int cy = screenH / 2;

        // رسم الدائرة في المنتصف
        Ellipse(hdc, cx - g_circleRadius, cy - g_circleRadius, cx + g_circleRadius, cy + g_circleRadius);

        SelectObject(hdc, oldPen); SelectObject(hdc, oldBrush);
        DeleteObject(hPen);
        EndPaint(hwnd, &ps);
        return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void StartOverlay() {
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = OverlayProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "AimbotOverlay";
    RegisterClass(&wc);

    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);

    HWND hwnd = CreateWindowEx(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "AimbotOverlay", "Overlay", WS_POPUP,
        0, 0, screenW, screenH, NULL, NULL, wc.hInstance, NULL
    );

    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY); // جعل الخلفية شفافة
    ShowWindow(hwnd, SW_SHOW);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
}

// =====================================================================
// التصويب والإطلاق
// =====================================================================
void SnapToHead(int yOffset) {
    POINT currentPos; GetCursorPos(&currentPos);
    int steps = 3 + (rand() % 3);
    int targetY = currentPos.y - yOffset;

    for (int i = 1; i <= steps; ++i) {
        float t = (float)i / steps;
        int nextY = currentPos.y + (int)((targetY - currentPos.y) * t);
        INPUT input = { 0 }; input.type = INPUT_MOUSE;
        input.mi.dy = nextY - currentPos.y; input.mi.dwFlags = MOUSEEVENTF_MOVE;
        SendInput(1, &input, sizeof(INPUT));
        std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
}

void FireWeapon() {
    INPUT click = { 0 }; click.type = INPUT_MOUSE; click.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &click, sizeof(INPUT));
    std::this_thread::sleep_for(std::chrono::milliseconds(25)); 
    click.mi.dwFlags = MOUSEEVENTF_LEFTUP; SendInput(1, &click, sizeof(INPUT));
}

// =====================================================================
// الدالة الرئيسية
// =====================================================================
int main() {
    std::cout << "[+] Starting Circular Triggerbot...\n";
    
    // تشغيل نافذة الدائرة في خيط منفصل
    std::thread overlayThread(StartOverlay);
    overlayThread.detach();
    
    DXGICaptureEngine capture;
    cv::Mat roiFrame;
    auto lastShotTime = std::chrono::high_resolution_clock::now();

    while (!capture.Init()) {
        std::cout << "[-] DXGI Init Failed. Retrying...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    std::cout << "[+] Engine Initialized. Hold [ALT] to trigger.\n";
    std::cout << "[+] Press [+] or [-] to resize the circle.\n";

    while (true) {
        // قراءة أزرار التحكم بحجم الدائرة
        if (GetAsyncKeyState(VK_ADD) & 0x8000 || GetAsyncKeyState(VK_OEM_PLUS) & 0x8000) {
            g_circleRadius += 2;
            if (g_circleRadius > 300) g_circleRadius = 300;
            InvalidateRect(FindWindow("AimbotOverlay", NULL), NULL, TRUE); // إعادة رسم الدائرة
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (GetAsyncKeyState(VK_SUBTRACT) & 0x8000 || GetAsyncKeyState(VK_OEM_MINUS) & 0x8000) {
            g_circleRadius -= 2;
            if (g_circleRadius < 10) g_circleRadius = 10;
            InvalidateRect(FindWindow("AimbotOverlay", NULL), NULL, TRUE);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        int status = capture.GrabFrame(roiFrame);
        if (status == -1) {
            capture.Release(); std::this_thread::sleep_for(std::chrono::milliseconds(500));
            while (!capture.Init()) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
            continue;
        }
        if (status == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

        if (status == 1 && !roiFrame.empty()) {
            cv::Mat hsvFrame;
            cv::cvtColor(roiFrame, hsvFrame, cv::COLOR_BGRA2BGR);
            cv::cvtColor(hsvFrame, hsvFrame, cv::COLOR_BGR2HSV);

            // إنشاء قناع دائري (Circular Mask)
            cv::Mat mask = cv::Mat::zeros(roiFrame.size(), CV_8UC1);
            cv::circle(mask, cv::Point(roiFrame.cols / 2, roiFrame.rows / 2), g_circleRadius, cv::Scalar(255), -1);

            // فلترة اللون الأحمر
            cv::Mat mask1, mask2, redMask, finalMask;
            cv::inRange(hsvFrame, cv::Scalar(0, 120, 100), cv::Scalar(10, 255, 255), mask1);
            cv::inRange(hsvFrame, cv::Scalar(170, 120, 100), cv::Scalar(180, 255, 255), mask2);
            cv::bitwise_or(mask1, mask2, redMask);

            // دمج قناع اللون الأحمر مع القناع الدائري (نتجاهل أي شيء خارج الدائرة)
            cv::bitwise_and(redMask, mask, finalMask);

            double redPixelCount = cv::countNonZero(finalMask);
            
            // إذا وجدنا بكسلات حمراء داخل الدائرة، وكان زر ALT مضغوطاً
            if (redPixelCount >= 10 && (GetAsyncKeyState(TRIGGER_KEY) & 0x8000)) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                float timeSinceLastShot = std::chrono::duration<float, std::milli>(currentTime - lastShotTime).count();
                
                if (timeSinceLastShot > 80.0f) {
                    lastShotTime = currentTime;
                    SnapToHead(15); 
                    FireWeapon();
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return 0;
}
