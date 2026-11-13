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
// Global Thread-Safe Variables
// =====================================================================
std::atomic<bool> g_appRunning(true);
std::atomic<int> g_circleRadius(120);
std::atomic<bool> g_isFiring(false);
std::atomic<bool> g_isAltPressed(false);
std::atomic<bool> g_needsRedraw(false);

std::atomic<int> g_overlayState(0); // 0=Init, 1=Success, -1=Failed
std::atomic<DWORD> g_overlayThreadId(0);
std::atomic<HWND> g_overlayHwnd(nullptr);
const int TRIGGER_KEY = VK_MENU; // ALT

std::vector<std::pair<cv::Mat, double>> g_templateBank;
auto lastRecoilTime = std::chrono::high_resolution_clock::now();
// =====================================================================

enum class CaptureStatus { Success, Timeout, AccessLost, Error, InvalidROI };

// =====================================================================
// RAII Wrappers (Strict Resource Safety)
// =====================================================================
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

// =====================================================================
// DXGI Capture Engine
// =====================================================================
class DXGICaptureEngine {
private:
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGIOutputDuplication> deskDupl;
    ComPtr<ID3D11Texture2D> stagingTex;
    
    int texWidth = 0; 
    int texHeight = 0;
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
        HRESULT hr = deskDupl->AcquireNextFrame(0, &frameInfo, desktopResource.GetAddressOf());
        
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
        
        // === الإصلاح الجذري: التحقق من FAILED أولاً ===
        if (FAILED(hr)) {
            return CaptureStatus::Error; 
        }

        // بما أن hr نجح، يجب علينا الآن ضمان استدعاء Unmap في كل الحالات
        DXGIMapLocker mapLocker(context.Get(), stagingTex.Get());

        // الآن يمكننا فحص المؤشر بأمان (إذا كان فارغاً، الـ RAII سيتكفل بالتنظيف)
        if (mapped.pData == nullptr) {
            return CaptureStatus::Error;
        }

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

// =====================================================================
// Input & Helper Functions
// =====================================================================
void SnapToTarget(int deltaY) {
    if (std::abs(deltaY) < 1) return;
    INPUT input = { 0 }; input.type = INPUT_MOUSE;
    input.mi.dy = deltaY; input.mi.dwFlags = MOUSEEVENTF_MOVE;
    SendInput(1, &input, sizeof(INPUT));
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

// =====================================================================
// Overlay Window
// =====================================================================
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
        MoveToEx(hdc, cx - 5, cy, NULL); LineTo(hdc, cx + 5, cy);
        MoveToEx(hdc, cx, cy - 5, NULL); LineTo(hdc, cx, cy + 5);
        SelectObject(hdc, oldPen); SelectObject(hdc, oldBrush); DeleteObject(hGreenPen);
        EndPaint(hwnd, &ps); return 0;
    }
    if (uMsg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

void OverlayThread() {
    g_overlayThreadId.store(GetCurrentThreadId());

    WNDCLASS wc = { 0 }; 
    wc.lpfnWndProc = OverlayProc; 
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "AimbotOverlay"; 
    
    if (!RegisterClass(&wc)) {
        g_overlayState.store(-1);
        return;
    }

    HWND hwnd = CreateWindowEx(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "AimbotOverlay", "Overlay", WS_POPUP, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), NULL, NULL, wc.hInstance, NULL);

    if (!hwnd) {
        UnregisterClass("AimbotOverlay", wc.hInstance);
        g_overlayState.store(-1);
        return;
    }

    g_overlayHwnd.store(hwnd);
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW);
    
    g_overlayState.store(1);

    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0) > 0) {
        TranslateMessage(&msg); DispatchMessage(&msg);
    }
    
    if (IsWindow(hwnd)) DestroyWindow(hwnd);
    UnregisterClass("AimbotOverlay", wc.hInstance);
}

BOOL WINAPI ConsoleHandler(DWORD signal) {
    if (signal == CTRL_C_EVENT || signal == CTRL_CLOSE_EVENT) {
        g_appRunning.store(false);
        
        HWND hwnd = g_overlayHwnd.load();
        if (hwnd && PostMessage(hwnd, WM_CLOSE, 0, 0)) {
            // WM_CLOSE سيؤدي إلى WM_DESTROY ثم WM_QUIT
        } else {
            DWORD threadId = g_overlayThreadId.load();
            if (threadId != 0) PostThreadMessage(threadId, WM_QUIT, 0, 0);
        }
        return TRUE;
    }
    return FALSE;
}

// =====================================================================
// Main Function
// =====================================================================
int main() {
    std::cout << "[+] Starting Ultimate Aimbot (V36.0)...\n";
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    
    cv::Mat rawTemplate = cv::imread("enemy_template.png", cv::IMREAD_GRAYSCALE);
    if (rawTemplate.empty()) {
        std::cerr << "[-] ERROR: Please place 'enemy_template.png' in the same folder!\n";
        return 1;
    }

    int cropHeight = (int)(rawTemplate.rows * 0.70);
    if (cropHeight > 10) rawTemplate = rawTemplate(cv::Rect(0, 0, rawTemplate.cols, cropHeight)).clone();
    
    for (double scale = 0.3; scale <= 1.8; scale += 0.1) {
        cv::Mat resized;
        cv::resize(rawTemplate, resized, cv::Size(), scale, scale);
        cv::Canny(resized, resized, 50, 150);
        g_templateBank.push_back({ resized, scale });
    }
    std::cout << "[+] Template Bank ready (" << g_templateBank.size() << " scales).\n";

    std::thread overlayTh(OverlayThread);
    
    while (g_overlayState.load() == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (g_overlayState.load() == -1) {
        std::cerr << "[-] FATAL: Failed to create Overlay window. Exiting.\n";
        if (overlayTh.joinable()) overlayTh.join();
        return 1;
    }
    std::cout << "[+] Overlay Initialized.\n";

    DXGICaptureEngine capture;
    int retryCount = 0;
    while (g_appRunning.load() && !capture.Init() && retryCount < 5) {
        std::cout << "[-] DXGI Init Failed. Retrying in 2s (" << (retryCount + 1) << "/5)...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));
        retryCount++;
    }

    if (!capture.IsInitialized()) {
        std::cerr << "[-] FATAL: Failed to initialize DXGI after 5 attempts. Exiting.\n";
        g_appRunning.store(false);
        
        HWND hwnd = g_overlayHwnd.load();
        if (hwnd && PostMessage(hwnd, WM_CLOSE, 0, 0)) {
            // Good
        } else {
            DWORD threadId = g_overlayThreadId.load();
            if (threadId != 0) PostThreadMessage(threadId, WM_QUIT, 0, 0);
        }
        
        if (overlayTh.joinable()) overlayTh.join();
        return 1;
    }

    std::cout << "[+] Engine Initialized. Hold [ALT] to aim.\n";
    auto lastInputCheck = std::chrono::high_resolution_clock::now();
    
    cv::Mat roiFrame, bgrFrame, grayFrame, hsvFrame, circleMask, searchEdges;
    std::vector<std::vector<cv::Point>> redContours;
    int lastMaskRadius = -1;
    
    auto lastPrintTime = std::chrono::high_resolution_clock::now();
    bool printedRedAnchor = false;

    while (g_appRunning.load()) {
        auto now = std::chrono::high_resolution_clock::now();
        
        if (std::chrono::duration<float, std::milli>(now - lastInputCheck).count() > 10.0f) {
            bool altState = (GetAsyncKeyState(TRIGGER_KEY) & 0x8000) != 0;
            g_isAltPressed.store(altState);
            if (!altState) SetFireState(false);

            int currentRadius = g_circleRadius.load();
            if (GetAsyncKeyState(VK_ADD) & 0x8000 || GetAsyncKeyState(VK_OEM_PLUS) & 0x8000) { g_circleRadius.store(std::min(currentRadius + 3, 500)); g_needsRedraw.store(true); }
            if (GetAsyncKeyState(VK_SUBTRACT) & 0x8000 || GetAsyncKeyState(VK_OEM_MINUS) & 0x8000) { g_circleRadius.store(std::max(currentRadius - 3, 30)); g_needsRedraw.store(true); }
            
            if (g_needsRedraw.exchange(false)) {
                HWND hwnd = g_overlayHwnd.load();
                if (hwnd) InvalidateRect(hwnd, NULL, TRUE);
            }
            lastInputCheck = now;
        }

        int currentRadius = g_circleRadius.load();
        CaptureStatus status = capture.GrabFrame(roiFrame, currentRadius);
        
        if (status == CaptureStatus::AccessLost) {
            capture.Cleanup(); 
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            int reinitRetries = 0;
            while (g_appRunning.load() && !capture.Init() && reinitRetries < 5) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                reinitRetries++;
            }
            if (!capture.IsInitialized()) {
                std::cerr << "[-] FATAL: Lost DXGI access and failed to recover. Exiting.\n";
                g_appRunning.store(false);
            }
            continue;
        } else if (status != CaptureStatus::Success) { 
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); 
            continue; 
        }

        if (!roiFrame.empty()) {
            if (capture.IsRGBA()) cv::cvtColor(roiFrame, bgrFrame, cv::COLOR_RGBA2BGR);
            else cv::cvtColor(roiFrame, bgrFrame, cv::COLOR_BGRA2BGR);

            cv::cvtColor(bgrFrame, grayFrame, cv::COLOR_BGR2GRAY);
            cv::cvtColor(bgrFrame, hsvFrame, cv::COLOR_BGR2HSV);
            cv::GaussianBlur(grayFrame, grayFrame, cv::Size(3, 3), 0);

            int cx = roiFrame.cols / 2; int cy = roiFrame.rows / 2;

            if (currentRadius != lastMaskRadius) {
                circleMask = cv::Mat::zeros(roiFrame.size(), CV_8UC1);
                cv::circle(circleMask, cv::Point(cx, cy), currentRadius, cv::Scalar(255), -1);
                lastMaskRadius = currentRadius;
            }

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

            if (anchorPoint.x != -1 && g_isAltPressed.load()) {
                if (std::chrono::duration<float, std::milli>(now - lastPrintTime).count() > 1000.0f) {
                    std::cout << "[DEBUG] Red Anchor Found. Searching shape...\n";
                    lastPrintTime = now;
                    printedRedAnchor = true;
                }

                int searchSize = currentRadius * 2;
                int sx = std::max(0, anchorPoint.x - searchSize / 2);
                int sy = std::max(0, anchorPoint.y - searchSize / 2);
                int sw = std::min(searchSize, roiFrame.cols - sx);
                int sh = std::min(searchSize, roiFrame.rows - sy);

                bool foundHead = false;
                int highestY = anchorPoint.y;
                double bestMatchValue = 0.0;

                if (sw > 0 && sh > 0) {
                    cv::Mat searchArea = grayFrame(cv::Rect(sx, sy, sw, sh));
                    cv::Canny(searchArea, searchEdges, 50, 150);

                    for (const auto& entry : g_templateBank) {
                        const cv::Mat& resizedTemplate = entry.first;
                        if (resizedTemplate.cols >= searchArea.cols || resizedTemplate.rows >= searchArea.rows) continue;

                        cv::Mat resultMat;
                        cv::matchTemplate(searchEdges, resizedTemplate, resultMat, cv::TM_CCOEFF_NORMED);

                        double minVal, maxVal;
                        cv::Point minLoc, maxLoc;
                        cv::minMaxLoc(resultMat, &minVal, &maxVal, &minLoc, &maxLoc);

                        if (maxVal > bestMatchValue) bestMatchValue = maxVal;

                        if (maxVal > 0.65) {
                            int matchCenterX = sx + maxLoc.x + (resizedTemplate.cols / 2);
                            int matchCenterY = sy + maxLoc.y + (resizedTemplate.rows / 2);
                            double dist = std::hypot(matchCenterX - cx, matchCenterY - cy);

                            if (dist <= currentRadius) {
                                highestY = sy + maxLoc.y;
                                foundHead = true;
                                break;
                            }
                        }
                    }
                }

                if (std::chrono::duration<float, std::milli>(now - lastPrintTime).count() > 1000.0f || (!foundHead && printedRedAnchor)) {
                    std::cout << "[DEBUG] Match Value: " << bestMatchValue << " | " << (foundHead ? "TARGET ACQUIRED!" : "Not recognized (0.65 required).") << "\n";
                    lastPrintTime = now;
                    printedRedAnchor = false;
                }

                if (foundHead) {
                    int deltaY = highestY - cy;
                    if (std::abs(deltaY) > 2 && std::abs(deltaY) < 150) SnapToTarget(deltaY);
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
                } else { SetFireState(false); }
            } else {
                if (g_isAltPressed.load() && std::chrono::duration<float, std::milli>(now - lastPrintTime).count() > 1000.0f) {
                    std::cout << "[DEBUG] No RED anchor found inside circle.\n";
                    lastPrintTime = now;
                }
                SetFireState(false); 
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "[+] Shutting down gracefully...\n";
    g_appRunning.store(false);
    
    HWND hwnd = g_overlayHwnd.load();
    if (hwnd && PostMessage(hwnd, WM_CLOSE, 0, 0)) {
        // Good
    } else {
        DWORD threadId = g_overlayThreadId.load();
        if (threadId != 0) PostThreadMessage(threadId, WM_QUIT, 0, 0);
    }
    
    if (overlayTh.joinable()) overlayTh.join();
    
    capture.Cleanup();
    std::cout << "[+] Clean exit.\n";
    return 0;
}
