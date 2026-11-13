#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include <mutex>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdi32.lib")

// ================= المتغيرات العامة =================
int g_circleRadius = 80; 
int g_headOffset = 0; 
const int TRIGGER_KEY = VK_MENU; // ALT
bool g_isFiring = false;
bool g_isAltPressed = false; 

// متغيرات لرسم المربع الأبيض حول "جسم اللاعب"
std::mutex g_boxMutex;
cv::Rect g_targetBox = cv::Rect(0, 0, 0, 0);
bool g_hasTarget = false;
// =====================================================

// نافذة الرسم (الدائرة الخضراء + المربع الأبيض حول اللاعب)
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_ERASEBKGND) return 1; 
    if (uMsg == WM_PAINT) {
        PAINTSTRUCT ps; HDC hdc = BeginPaint(hwnd, &ps);
        RECT rect; GetClientRect(hwnd, &rect);
        HBRUSH bgBrush = CreateSolidBrush(RGB(0, 0, 0)); 
        FillRect(hdc, &rect, bgBrush); DeleteObject(bgBrush);

        int screenW = GetSystemMetrics(SM_CXSCREEN); int screenH = GetSystemMetrics(SM_CYSCREEN);
        int cx = screenW / 2; int cy = screenH / 2;

        // 1. رسم الدائرة الخضراء
        HPEN hGreenPen = CreatePen(PS_SOLID, 2, RGB(0, 255, 0));
        HGDIOBJ oldPen = SelectObject(hdc, hGreenPen);
        HBRUSH hNullBrush = (HBRUSH)GetStockObject(NULL_BRUSH); 
        HGDIOBJ oldBrush = SelectObject(hdc, hNullBrush);
        
        Ellipse(hdc, cx - g_circleRadius, cy - g_circleRadius, cx + g_circleRadius, cy + g_circleRadius);
        MoveToEx(hdc, cx - 5, cy, NULL); LineTo(hdc, cx + 5, cy);
        MoveToEx(hdc, cx, cy - 5, NULL); LineTo(hdc, cx, cy + 5);

        // 2. رسم المربع الأبيض حول "اللاعب"
        {
            std::lock_guard<std::mutex> lock(g_boxMutex);
            if (g_hasTarget && g_targetBox.width > 0) {
                HPEN hWhitePen = CreatePen(PS_SOLID, 2, RGB(255, 255, 255));
                SelectObject(hdc, hWhitePen);
                
                int boxX = (cx - g_circleRadius) + g_targetBox.x;
                int boxY = (cy - g_circleRadius) + g_targetBox.y;
                
                // رسم المربع حول جسم اللاعب (وليس حول المؤشر)
                Rectangle(hdc, boxX, boxY, boxX + g_targetBox.width, boxY + g_targetBox.height);
                DeleteObject(hWhitePen);
            }
        }

        SelectObject(hdc, oldPen); SelectObject(hdc, oldBrush); DeleteObject(hGreenPen);
        EndPaint(hwnd, &ps); return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void StartOverlay() {
    WNDCLASS wc = { 0 }; wc.lpfnWndProc = OverlayProc; wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "AimbotOverlay"; RegisterClass(&wc);
    int screenW = GetSystemMetrics(SM_CXSCREEN); int screenH = GetSystemMetrics(SM_CYSCREEN);
    HWND hwnd = CreateWindowEx(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "AimbotOverlay", "Overlay", WS_POPUP, 0, 0, screenW, screenH, NULL, NULL, wc.hInstance, NULL);
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY); ShowWindow(hwnd, SW_SHOW);
    MSG msg; while (GetMessage(&msg, NULL, 0, 0)) { TranslateMessage(&msg); DispatchMessage(&msg); }
}

// محرك الالتقاط
class DXGICaptureEngine {
private:
    ID3D11Device* device = nullptr; ID3D11DeviceContext* context = nullptr;
    IDXGIOutputDuplication* deskDupl = nullptr; ID3D11Texture2D* stagingTex = nullptr;
    int texWidth = 0; int texHeight = 0;
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
            device->CreateTexture2D(&desc, nullptr, &stagingTex); texWidth = desc.Width; texHeight = desc.Height;
        }
        context->CopyResource(stagingTex, acquiredDesktopImage);
        D3D11_MAPPED_SUBRESOURCE mapped; hr = context->Map(stagingTex, 0, D3D11_MAP_READ, 0, &mapped);
        if (SUCCEEDED(hr)) {
            cv::Mat fullFrame(desc.Height, desc.Width, CV_8UC4, mapped.pData, mapped.RowPitch);
            int centerX = desc.Width / 2; int centerY = desc.Height / 2; int FOV = g_circleRadius + 20; 
            cv::Rect roi(centerX - FOV, centerY - FOV, FOV * 2, FOV * 2);
            if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= desc.Width && roi.y + roi.height <= desc.Height) outFrame = fullFrame(roi).clone();
            context->Unmap(stagingTex, 0);
        }
        acquiredDesktopImage->Release(); deskDupl->ReleaseFrame(); return 1;
    }
    void Release() {
        if (stagingTex) stagingTex->Release(); if (deskDupl) deskDupl->Release();
        if (context) context->Release(); if (device) device->Release();
    }
};

// التصويب
void SnapToTarget(int deltaX, int deltaY) {
    INPUT input = { 0 }; input.type = INPUT_MOUSE;
    input.mi.dx = deltaX; input.mi.dy = deltaY; input.mi.dwFlags = MOUSEEVENTF_MOVE;
    SendInput(1, &input, sizeof(INPUT));
}

void SetFireState(bool fire) {
    if (fire && !g_isFiring) {
        INPUT click = { 0 }; click.type = INPUT_MOUSE; click.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &click, sizeof(INPUT)); g_isFiring = true;
    } else if (!fire && g_isFiring) {
        INPUT click = { 0 }; click.type = INPUT_MOUSE; click.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &click, sizeof(INPUT)); g_isFiring = false;
    }
}

int main() {
    std::cout << "[+] Starting Final Visual Aimbot (V13.0)...\n";
    std::thread overlayThread(StartOverlay); overlayThread.detach();
    
    DXGICaptureEngine capture; cv::Mat roiFrame;
    while (!capture.Init()) { std::this_thread::sleep_for(std::chrono::seconds(2)); }
    std::cout << "[+] Engine Initialized. Hold [ALT] to aim.\n";

    auto lastInputCheck = std::chrono::high_resolution_clock::now();
    HWND overlayHwnd = NULL;

    while (true) {
        if (!overlayHwnd) overlayHwnd = FindWindow("AimbotOverlay", NULL);

        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration<float, std::milli>(now - lastInputCheck).count() > 15.0f) {
            g_isAltPressed = (GetAsyncKeyState(TRIGGER_KEY) & 0x8000) != 0;
            
            if (GetAsyncKeyState(VK_ADD) & 0x8000 || GetAsyncKeyState(VK_OEM_PLUS) & 0x8000 || GetAsyncKeyState(VK_OEM_6) & 0x8000) { g_circleRadius += 3; }
            if (GetAsyncKeyState(VK_SUBTRACT) & 0x8000 || GetAsyncKeyState(VK_OEM_MINUS) & 0x8000 || GetAsyncKeyState(VK_OEM_4) & 0x8000) { g_circleRadius -= 3; if(g_circleRadius < 20) g_circleRadius = 20; }
            if (GetAsyncKeyState(VK_UP) & 0x8000) { g_headOffset++; }
            if (GetAsyncKeyState(VK_DOWN) & 0x8000) { g_headOffset--; if(g_headOffset < -30) g_headOffset = -30; }
            
            lastInputCheck = now;
        }

        int status = capture.GrabFrame(roiFrame);
        if (status == -1) {
            capture.Release(); std::this_thread::sleep_for(std::chrono::milliseconds(500));
            while (!capture.Init()) { std::this_thread::sleep_for(std::chrono::seconds(1)); } continue;
        }
        if (status == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

        if (status == 1 && !roiFrame.empty()) {
            cv::Mat bgrFrame, hsvFrame;
            cv::cvtColor(roiFrame, bgrFrame, cv::COLOR_BGRA2BGR);
            cv::cvtColor(bgrFrame, hsvFrame, cv::COLOR_BGR2HSV);

            int cx = roiFrame.cols / 2;
            int cy = roiFrame.rows / 2;

            // 1. التحقق من "المؤشر" (هل المنتصف أحمر؟)
            cv::Vec3b centerPixel = hsvFrame.at<cv::Vec3b>(cy, cx);
            int H = centerPixel[0], S = centerPixel[1], V = centerPixel[2];
            bool isTargeting = ((H >= 0 && H <= 10) || (H >= 170 && H <= 180)) && (S > 120) && (V > 100);

            bool foundBody = false;
            int highestY = cy;
            cv::Rect detectedBox;

            // 2. إذا كان المؤشر أحمر، نبحث عن "جسم" (بكسلات مختلفة عن الخلفية)
            if (isTargeting) {
                cv::Rect bgPatch(cx - g_circleRadius + 5, cy - g_circleRadius + 5, 10, 10);
                if (bgPatch.x >= 0 && bgPatch.y >= 0 && bgPatch.x + bgPatch.width <= roiFrame.cols && bgPatch.y + bgPatch.height <= roiFrame.rows) {
                    cv::Scalar bgColor = cv::mean(bgrFrame(bgPatch));

                    cv::Mat diffMask;
                    cv::absdiff(bgrFrame, bgColor, diffMask);
                    cv::Mat grayDiff;
                    cv::cvtColor(diffMask, grayDiff, cv::COLOR_BGR2GRAY);
                    
                    cv::Mat colorThreshold;
                    cv::threshold(grayDiff, colorThreshold, 40, 255, cv::THRESH_BINARY);

                    cv::Mat circleMask = cv::Mat::zeros(roiFrame.size(), CV_8UC1);
                    cv::circle(circleMask, cv::Point(cx, cy), g_circleRadius, cv::Scalar(255), -1);
                    cv::bitwise_and(colorThreshold, circleMask, colorThreshold);

                    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
                    cv::morphologyEx(colorThreshold, colorThreshold, cv::MORPH_CLOSE, kernel);

                    std::vector<std::vector<cv::Point>> contours;
                    cv::findContours(colorThreshold, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                    for (const auto& cnt : contours) {
                        double area = cv::contourArea(cnt);
                        if (area > 100) { 
                            cv::Rect rect = cv::boundingRect(cnt);
                            
                            // 3. شرط الطول > العرض (للتأكد أنه إنسان)
                            if (rect.height > rect.width * 1.2) {
                                // هل هذا الجسم يمر عبر المنتصف؟
                                if (rect.contains(cv::Point(cx, cy))) {
                                    detectedBox = rect;
                                    highestY = rect.y; 
                                    foundBody = true;
                                    break;
                                }
                            }
                        }
                    }
                }
            }

            // 4. تحديث الرسم (نرسم المربع حول الجسم، وليس حول المؤشر)
            {
                std::lock_guard<std::mutex> lock(g_boxMutex);
                if (foundBody) {
                    g_targetBox = detectedBox;
                    g_hasTarget = true;
                } else {
                    g_hasTarget = false;
                }
            }
            
            if (overlayHwnd) InvalidateRect(overlayHwnd, NULL, TRUE);

            // 5. التصويب
            if (foundBody && g_isAltPressed) {
                int targetY = highestY + g_headOffset;
                int deltaY = targetY - cy;
                
                if (std::abs(deltaY) > 2) {
                    SnapToTarget(0, deltaY);
                }
                SetFireState(true);
            } else {
                SetFireState(false);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return 0;
}
