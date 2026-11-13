#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include <atomic>
#include <algorithm>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdi32.lib")

using Microsoft::WRL::ComPtr;

// =====================================================================
// Global Variables & Settings
// =====================================================================
std::atomic<bool> g_appRunning(true);
std::atomic<int> g_circleRadius(120);
std::atomic<bool> g_isFiring(false); 
std::atomic<bool> g_isAltPressed(false);
std::atomic<bool> g_needsRedraw(false);
std::atomic<int> g_overlayState(0); 
std::atomic<DWORD> g_overlayThreadId(0);
std::atomic<HWND> g_overlayHwnd(nullptr);
const int TRIGGER_KEY = VK_MENU; // ALT

cv::Mat g_refTemplate; // الصورة المرجعية (بدون تشويه)

auto lastRecoilTime = std::chrono::high_resolution_clock::now();
auto lastTimeTargetSeen = std::chrono::high_resolution_clock::now(); 

std::atomic<double> g_matchThreshold(0.35); 
std::atomic<int> g_headOffset(0);           
// =====================================================================

enum class CaptureStatus { Success, Timeout, AccessLost, Error, InvalidROI };

class DXGIFrameLocker {
private:
    IDXGIOutputDuplication* duplication;
public:
    DXGIFrameLocker(IDXGIOutputDuplication* dup) : duplication(dup) {}
    ~DXGIFrameLocker() { if (duplication) duplication->ReleaseFrame(); }
};

class DXGIMapLocker {
private:
    ID3D11DeviceContext* context;
    ID3D11Resource* resource;
public:
    DXGIMapLocker(ID3D11DeviceContext* ctx, ID3D11Resource* res) : context(ctx), resource(res) {}
    ~DXGIMapLocker() { if (context && resource) context->Unmap(resource, 0); }
};

class DXGICaptureEngine {
private:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGIOutputDuplication> deskDupl;
    ComPtr<ID3D11Texture2D> stagingTex;
    int texWidth = 0; int texHeight = 0;
    DXGI_FORMAT texFormat = DXGI_FORMAT_UNKNOWN; 
    bool isInitialized = false;
    bool isRGBA = false;

public:
    bool IsInitialized() const { return isInitialized; }

    bool Init() {
        Cleanup(); 
        D3D_FEATURE_LEVEL featureLevel;
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, device.GetAddressOf(), &featureLevel, context.GetAddressOf());
        if (FAILED(hr)) return false;

        ComPtr<IDXGIDevice> dxgiDevice; hr = device.As(&dxgiDevice); if (FAILED(hr)) return false;
        ComPtr<IDXGIAdapter> dxgiAdapter; hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)dxgiAdapter.GetAddressOf()); if (FAILED(hr)) return false;
        ComPtr<IDXGIOutput> dxgiOutput; hr = dxgiAdapter->EnumOutputs(0, dxgiOutput.GetAddressOf()); if (FAILED(hr)) return false;
        ComPtr<IDXGIOutput1> dxgiOutput1; hr = dxgiOutput.As(&dxgiOutput1); if (FAILED(hr)) return false;

        hr = dxgiOutput1->DuplicateOutput(device.Get(), deskDupl.GetAddressOf());
        if (SUCCEEDED(hr)) { isInitialized = true; return true; }
        return false;
    }

    bool IsRGBA() const { return isRGBA; }

    CaptureStatus GrabFrame(cv::Mat& outFrame, int currentRadius) {
        if (!isInitialized || !deskDupl) return CaptureStatus::Error;
        outFrame.release(); 

        DXGI_OUTDUPL_FRAME_INFO frameInfo = { 0 };
        ComPtr<IDXGIResource> desktopResource;
        
        HRESULT hr = deskDupl->AcquireNextFrame(10, &frameInfo, desktopResource.GetAddressOf());
        
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return CaptureStatus::Timeout;
        if (hr == DXGI_ERROR_ACCESS_LOST) return CaptureStatus::AccessLost;
        if (FAILED(hr)) return CaptureStatus::Error;

        DXGIFrameLocker frameLocker(deskDupl.Get());

        ComPtr<ID3D11Texture2D> acquiredImage;
        hr = desktopResource.As(&acquiredImage);
        if (FAILED(hr)) return CaptureStatus::Error;

        D3D11_TEXTURE2D_DESC desc;
        acquiredImage->GetDesc(&desc);

        if (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) isRGBA = true;
        else if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM) isRGBA = false;
        else return CaptureStatus::Error;

        if (!stagingTex || texWidth != (int)desc.Width || texHeight != (int)desc.Height || texFormat != desc.Format) {
            stagingTex.Reset();
            desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.BindFlags = 0; desc.MiscFlags = 0;
            hr = device->CreateTexture2D(&desc, nullptr, stagingTex.GetAddressOf());
            if (FAILED(hr)) return CaptureStatus::Error;
            texWidth = desc.Width; texHeight = desc.Height; texFormat = desc.Format;
        }

        context->CopyResource(stagingTex.Get(), acquiredImage.Get());

        D3D11_MAPPED_SUBRESOURCE mapped = { 0 };
        hr = context->Map(stagingTex.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) return CaptureStatus::Error; 

        DXGIMapLocker mapLocker(context.Get(), stagingTex.Get());
        if (mapped.pData == nullptr) return CaptureStatus::Error;

        int centerX = desc.Width / 2; int centerY = desc.Height / 2;
        int fov = currentRadius + 20;
        cv::Rect roi(centerX - fov, centerY - fov, fov * 2, fov * 2);
        
        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > (int)desc.Width || roi.y + roi.height > (int)desc.Height) {
            return CaptureStatus::InvalidROI;
        }

        cv::Mat fullFrame(desc.Height, desc.Width, CV_8UC4, mapped.pData, mapped.RowPitch);
        outFrame = fullFrame(roi).clone();
        return CaptureStatus::Success;
    }

    void Cleanup() {
        isInitialized = false;
        stagingTex.Reset(); deskDupl.Reset(); context.Reset(); device.Reset();
        texWidth = 0; texHeight = 0; texFormat = DXGI_FORMAT_UNKNOWN;
    }
};

void SmoothSnapToTarget(int deltaX, int deltaY) {
    if (std::abs(deltaX) < 2 && std::abs(deltaY) < 2) return;
    int steps = 3; 
    int stepX = deltaX / steps;
    int stepY = deltaY / steps;
    
    for (int i = 0; i < steps; i++) {
        INPUT input = { 0 }; input.type = INPUT_MOUSE;
        input.mi.dx = (i == steps - 1) ? (deltaX - stepX * (steps - 1)) : stepX; 
        input.mi.dy = (i == steps - 1) ? (deltaY - stepY * (steps - 1)) : stepY; 
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        SendInput(1, &input, sizeof(INPUT));
        std::this_thread::sleep_for(std::chrono::microseconds(300)); 
    }
}

void SetFireState(bool fire) {
    if (fire && !g_isFiring.load()) {
        INPUT click = { 0 }; click.type = INPUT_MOUSE; click.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &click, sizeof(INPUT)); g_isFiring.store(true);
    } else if (!fire && g_isFiring.load()) {
        INPUT click = { 0 }; click.type = INPUT_MOUSE; click.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &click, sizeof(INPUT)); g_isFiring.store(false);
    }
}

LRESULT CALLBACK OverlayProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    if (uMsg == WM_ERASEBKGND) return 1;
    if (uMsg == WM_PAINT) {
        PAINTSTRUCT ps; HDC hdc = BeginPaint(hwnd, &ps);
        RECT rect; GetClientRect(hwnd, &rect);
        HBRUSH bgBrush = CreateSolidBrush(RGB(0, 0, 0)); FillRect(hdc, &rect, bgBrush); DeleteObject(bgBrush);
        int cx = GetSystemMetrics(SM_CXSCREEN) / 2; int cy = GetSystemMetrics(SM_CYSCREEN) / 2;
        int radius = g_circleRadius.load();
        HPEN hGreenPen = CreatePen(PS_SOLID, 2, RGB(0, 255, 0));
        HGDIOBJ oldPen = SelectObject(hdc, hGreenPen);
        HBRUSH hNullBrush = (HBRUSH)GetStockObject(NULL_BRUSH);
        HGDIOBJ oldBrush = SelectObject(hdc, hNullBrush);
        Ellipse(hdc, cx - radius, cy - radius, cx + radius, cy + radius);
        SelectObject(hdc, oldPen); SelectObject(hdc, oldBrush); DeleteObject(hGreenPen);
        EndPaint(hwnd, &ps); return 0;
    }
    if (uMsg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void OverlayThread() {
    g_overlayThreadId.store(GetCurrentThreadId());
    WNDCLASS wc = { 0 }; wc.lpfnWndProc = OverlayProc; wc.hInstance = GetModuleHandle(NULL); wc.lpszClassName = "AimbotOverlay"; 
    if (!RegisterClass(&wc)) { g_overlayState.store(-1); return; }
    HWND hwnd = CreateWindowEx(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "AimbotOverlay", "Overlay", WS_POPUP, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), NULL, NULL, wc.hInstance, NULL);
    if (!hwnd) { UnregisterClass("AimbotOverlay", wc.hInstance); g_overlayState.store(-1); return; }
    g_overlayHwnd.store(hwnd);
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW);
    g_overlayState.store(1);
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessage(&msg); }
    if (IsWindow(hwnd)) DestroyWindow(hwnd);
    UnregisterClass("AimbotOverlay", wc.hInstance);
}

BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT) {
        g_appRunning.store(false);
        HWND hwnd = g_overlayHwnd.load();
        if (hwnd && PostMessage(hwnd, WM_CLOSE, 0, 0)) {}
        else {
            DWORD threadId = g_overlayThreadId.load();
            if (threadId != 0) PostThreadMessage(threadId, WM_QUIT, 0, 0);
        }
        return TRUE;
    }
    return FALSE;
}

int main() {
    std::cout << "[+] Starting True Multi-Scale Aimbot (V60.0)...\n";
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    
    cv::Mat rawTemplate = cv::imread("enemy_template.png", cv::IMREAD_GRAYSCALE);
    if (rawTemplate.empty()) {
        std::cerr << "[-] FATAL ERROR: Could not load 'enemy_template.png'.\n";
        return 1;
    }
    
    cv::GaussianBlur(rawTemplate, g_refTemplate, cv::Size(5, 5), 0);
    cv::Canny(g_refTemplate, g_refTemplate, 50, 150);
    cv::dilate(g_refTemplate, g_refTemplate, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    std::thread overlayTh(OverlayThread);
    while (g_overlayState.load() == 0) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); }
    if (g_overlayState.load() == -1) { std::cerr << "[-] FATAL ERROR: Overlay Failed.\n"; if (overlayTh.joinable()) overlayTh.join(); return 1; }

    DXGICaptureEngine capture;
    while (g_appRunning.load() && !capture.Init()) { std::this_thread::sleep_for(std::chrono::seconds(2)); }
    if (!capture.IsInitialized()) { g_appRunning.store(false); if (overlayTh.joinable()) overlayTh.join(); return 1; }

    std::cout << "[+] Engine Initialized Successfully. Hold [ALT] to aim.\n";
    
    auto lastInputCheck = std::chrono::high_resolution_clock::now();
    auto lastRadiusChange = std::chrono::high_resolution_clock::now(); 
    
    cv::Mat roiFrame, bgrFrame, grayFrame, hsvFrame, circleMask, searchEdges;
    std::vector<std::vector<cv::Point>> redContours;
    int lastMaskRadius = -1;
    auto lastPrintTime = std::chrono::high_resolution_clock::now();

    while (g_appRunning.load()) {
        auto now = std::chrono::high_resolution_clock::now();
        
        if (std::chrono::duration<float, std::milli>(now - lastInputCheck).count() > 20.0f) {
            bool altState = (GetAsyncKeyState(TRIGGER_KEY) & 0x8000) != 0;
            g_isAltPressed.store(altState);
            
            if (!altState) SetFireState(false);

            int currentRadius = g_circleRadius.load();
            if (GetAsyncKeyState(VK_ADD) & 0x8000 || GetAsyncKeyState(VK_OEM_PLUS) & 0x8000) { 
                if (std::chrono::duration<float, std::milli>(now - lastRadiusChange).count() > 50.0f) {
                    g_circleRadius.store(std::min(currentRadius + 3, 500)); g_needsRedraw.store(true); lastRadiusChange = now;
                }
            }
            if (GetAsyncKeyState(VK_SUBTRACT) & 0x8000 || GetAsyncKeyState(VK_OEM_MINUS) & 0x8000) { 
                if (std::chrono::duration<float, std::milli>(now - lastRadiusChange).count() > 50.0f) {
                    g_circleRadius.store(std::max(currentRadius - 3, 30)); g_needsRedraw.store(true); lastRadiusChange = now;
                }
            }
            
            if (GetAsyncKeyState(VK_LEFT) & 0x8000) { g_matchThreshold.store(std::max(g_matchThreshold.load() - 0.01, 0.10)); std::cout << "[CONTROL] Threshold: " << g_matchThreshold.load() << "\n"; std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
            if (GetAsyncKeyState(VK_RIGHT) & 0x8000) { g_matchThreshold.store(std::min(g_matchThreshold.load() + 0.01, 0.80)); std::cout << "[CONTROL] Threshold: " << g_matchThreshold.load() << "\n"; std::this_thread::sleep_for(std::chrono::milliseconds(100)); }
            
            if (GetAsyncKeyState(VK_UP) & 0x8000) { g_headOffset.store(g_headOffset.load() + 1); std::cout << "[CONTROL] Head Offset: " << g_headOffset.load() << "\n"; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
            if (GetAsyncKeyState(VK_DOWN) & 0x8000) { g_headOffset.store(g_headOffset.load() - 1); std::cout << "[CONTROL] Head Offset: " << g_headOffset.load() << "\n"; std::this_thread::sleep_for(std::chrono::milliseconds(50)); }
            
            if (g_needsRedraw.exchange(false)) {
                HWND hwnd = g_overlayHwnd.load();
                if (hwnd) InvalidateRect(hwnd, NULL, TRUE);
            }
            lastInputCheck = now;
        }

        int currentRadius = g_circleRadius.load();
        CaptureStatus status = capture.GrabFrame(roiFrame, currentRadius);
        
        if (status == CaptureStatus::AccessLost) {
            capture.Cleanup(); std::this_thread::sleep_for(std::chrono::milliseconds(500));
            while (g_appRunning.load() && !capture.Init()) { std::this_thread::sleep_for(std::chrono::seconds(1)); }
            continue;
        } else if (status == CaptureStatus::Timeout) {
            continue; 
        } else if (status != CaptureStatus::Success) { 
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); continue; 
        }

        if (!roiFrame.empty()) {
            if (capture.IsRGBA()) cv::cvtColor(roiFrame, bgrFrame, cv::COLOR_RGBA2BGR);
            else cv::cvtColor(roiFrame, bgrFrame, cv::COLOR_BGRA2BGR);

            cv::cvtColor(bgrFrame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::cvtColor(bgrFrame, hsvFrame, cv::COLOR_BGR2HSV);
            cv::GaussianBlur(grayFrame, grayFrame, cv::Size(5, 5), 0);

            int cx = roiFrame.cols / 2; int cy = roiFrame.rows / 2;

            if (currentRadius != lastMaskRadius) {
                circleMask = cv::Mat::zeros(roiFrame.size(), CV_8UC1);
                cv::circle(circleMask, cv::Point(cx, cy), currentRadius, cv::Scalar(255), -1);
                lastMaskRadius = currentRadius;
            }

            // البحث عن المؤشر الأحمر
            cv::Mat mask1, mask2, redMask, finalRedMask;
            cv::inRange(hsvFrame, cv::Scalar(0, 120, 100), cv::Scalar(10, 255, 255), mask1);
            cv::inRange(hsvFrame, cv::Scalar(170, 120, 100), cv::Scalar(180, 255, 255), mask2);
            cv::bitwise_or(mask1, mask2, redMask);
            cv::bitwise_and(redMask, circleMask, finalRedMask);

            redContours.clear();
            cv::findContours(finalRedMask, redContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            cv::Point anchorPoint(-1, -1);
            double minDist = 99999;

            for (const auto& cnt : redContours) {
                double area = cv::contourArea(cnt);
                if (area > 5) {
                    cv::Rect rect = cv::boundingRect(cnt);
                    cv::Point currentCenter(rect.x + rect.width / 2, rect.y + rect.height / 2);
                    double dist = std::hypot(currentCenter.x - cx, currentCenter.y - cy);
                    if (dist < minDist) { minDist = dist; anchorPoint = currentCenter; }
                }
            }

            // المطابقة الذكية
            if (anchorPoint.x != -1 && g_isAltPressed.load()) {
                cv::Mat searchEdges;
                cv::Canny(grayFrame, searchEdges, 40, 120);
                cv::dilate(searchEdges, searchEdges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));
                cv::bitwise_and(searchEdges, circleMask, searchEdges); 

                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(searchEdges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                bool foundHead = false;
                int targetX = cx;
                int targetY = cy;
                double bestScore = 0.0;
                double minDistanceToRed = 99999.0;
                double threshold = g_matchThreshold.load();
                bool foundAnyShape = false;
                std::string lastShapeInfo = "";

                for (const auto& cnt : contours) {
                    cv::Rect rect = cv::boundingRect(cnt);
                    
                    if (rect.width < 20 || rect.height < 40) continue; 
                    double aspect = (double)rect.height / rect.width;
                    if (aspect < 1.2 || aspect > 5.0) continue; 

                    double distToRed = std::hypot((rect.x + rect.width/2) - anchorPoint.x, (rect.y + rect.height/2) - anchorPoint.y);
                    if (distToRed > currentRadius) continue;

                    foundAnyShape = true;

                    // === الإصلاح الجذري: أخذ مساحة أكبر قليلاً حول الشكل ===
                    int padding = 10;
                    int padX = std::max(0, rect.x - padding);
                    int padY = std::max(0, rect.y - padding);
                    int padW = std::min(searchEdges.cols - padX, rect.width + padding * 2);
                    int padH = std::min(searchEdges.rows - padY, rect.height + padding * 2);
                    
                    cv::Mat candidateArea = searchEdges(cv::Rect(padX, padY, padW, padH));

                    // === الإصلاح الجذري: تجربة أحجام متعددة دون تشويه القالب ===
                    for (double scale = 0.5; scale <= 1.5; scale += 0.1) {
                        cv::Mat resizedRef;
                        // تغيير الحجم مع الحفاظ على النسبة الأصلية
                        cv::resize(g_refTemplate, resizedRef, cv::Size(), scale, scale);

                        // تجاهل إذا كان القالب أكبر من مساحة البحث
                        if (resizedRef.cols >= candidateArea.cols || resizedRef.rows >= candidateArea.rows) continue;

                        cv::Mat result;
                        cv::matchTemplate(candidateArea, resizedRef, result, cv::TM_CCOEFF_NORMED);
                        double minVal, maxVal;
                        cv::Point minLoc, maxLoc;
                        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

                        if (maxVal > bestScore) {
                            bestScore = maxVal;
                            lastShapeInfo = "Shape (W:" + std::to_string(rect.width) + ", H:" + std::to_string(rect.height) + ") Scale: " + std::to_string(scale).substr(0,4) + " Match: " + std::to_string((int)(maxVal*100)) + "%";

                            if (maxVal > threshold) {
                                if (distToRed < minDistanceToRed) {
                                    minDistanceToRed = distToRed;
                                    // التصويب على منتصف أعلى نقطة في المربع الأصلي
                                    targetX = rect.x + (rect.width / 2);
                                    targetY = rect.y;
                                    foundHead = true;
                                }
                            }
                        }
                    }
                }

                if (foundHead) {
                    lastTimeTargetSeen = now; 
                    int deltaX = targetX - cx;
                    int deltaY = (targetY - cy) + g_headOffset.load();
                    
                    if (std::abs(deltaX) > 2 || std::abs(deltaY) > 2) SmoothSnapToTarget(deltaX, deltaY); 
                    SetFireState(true); 

                    if (g_isFiring.load()) {
                        auto currentTime = std::chrono::high_resolution_clock::now();
                        if (std::chrono::duration<float, std::milli>(currentTime - lastRecoilTime).count() > 30.0f) {
                            INPUT recoilInput = { 0 };
                            recoilInput.type = INPUT_MOUSE; recoilInput.mi.dy = 2;
                            recoilInput.mi.dwFlags = MOUSEEVENTF_MOVE;
                            SendInput(1, &recoilInput, sizeof(INPUT));
                            lastRecoilTime = currentTime;
                        }
                    }

                    if (std::chrono::duration<float, std::milli>(now - lastPrintTime).count() > 1000.0f) {
                        std::cout << "[SUCCESS] Best: " << lastShapeInfo << " -> TARGET ACQUIRED!\n";
                        lastPrintTime = now;
                    }
                } else { 
                    auto timeSinceSeen = std::chrono::duration<float, std::milli>(now - lastTimeTargetSeen).count();
                    if (timeSinceSeen > 150.0f) {
                        SetFireState(false); 
                    }

                    if (std::chrono::duration<float, std::milli>(now - lastPrintTime).count() > 1000.0f) {
                        if (foundAnyShape) {
                            std::cout << "[DEBUG] " << lastShapeInfo << " -> IGNORED (Below Threshold " << threshold << ")\n";
                        } else {
                            std::cout << "[DEBUG] Searched inside circle. No shape matching human proportions found.\n";
                        }
                        lastPrintTime = now;
                    }
                }
            } else {
                if (g_isAltPressed.load() && std::chrono::duration<float, std::milli>(now - lastPrintTime).count() > 1000.0f) {
                    std::cout << "[DEBUG] Searching... No RED anchor found inside the circle.\n";
                    lastPrintTime = now;
                }
                SetFireState(false); 
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    g_appRunning.store(false);
    HWND hwnd = g_overlayHwnd.load();
    if (hwnd && PostMessage(hwnd, WM_CLOSE, 0, 0)) {} else {
        DWORD threadId = g_overlayThreadId.load();
        if (threadId != 0) PostThreadMessage(threadId, WM_QUIT, 0, 0);
    }
    if (overlayTh.joinable()) overlayTh.join();
    capture.Cleanup();
    return 0;
}
