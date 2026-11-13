// language: C++, file: triggerbot_v62.cpp, runtime: MSVC v143 (C++17), Windows 10/11
// Red-only triggerbot — holds LMB while ALT + red in circle. Auto-lift by liftValue px every liftIntervalMs.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include <atomic>
#include <algorithm>
#include <mutex>
#include <map>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

// =====================================================================
//  CONFIG
// =====================================================================
struct Config {
    int circleRadius   = 120;
    int liftValue      = 3;   // pixels up per tick
    int liftIntervalMs = 30;  // tick interval while firing
    int fireGraceMs    = 80;  // red absent this long -> release
    int minRedArea     = 5;
};

static Config LoadConfig(const char* path) {
    Config c;
    std::ifstream f(path);
    if (!f) return c;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq);
        std::string v = line.substr(eq + 1);
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
        try {
            if      (k == "circleRadius")   c.circleRadius   = std::stoi(v);
            else if (k == "liftValue")      c.liftValue      = std::stoi(v);
            else if (k == "liftIntervalMs") c.liftIntervalMs = std::stoi(v);
            else if (k == "fireGraceMs")    c.fireGraceMs    = std::stoi(v);
            else if (k == "minRedArea")     c.minRedArea     = std::stoi(v);
        } catch (...) {}
    }
    return c;
}

// =====================================================================
//  GLOBALS
// =====================================================================
static std::atomic<bool>  g_appRunning(true);
static std::atomic<int>   g_circleRadius(120);
static std::atomic<int>   g_liftValue(3);
static std::atomic<int>   g_liftIntervalMs(30);
static std::atomic<int>   g_minRedArea(5);

static std::atomic<bool>  g_isFiring(false);
static std::atomic<bool>  g_isAltPressed(false);
static std::atomic<bool>  g_redVisible(false);
static std::atomic<DWORD> g_overlayThreadId(0);
static std::atomic<HWND>  g_overlayHwnd(nullptr);
static std::atomic<int>   g_overlayState(0);
static std::atomic<float> g_fps(0.0f);

static const int TRIGGER_KEY = VK_MENU; // ALT

struct OverlayStatus {
    bool  red     = false;
    bool  alt     = false;
    bool  firing  = false;
    int   radius  = 120;
    int   lift    = 3;
    float fps     = 0.0f;
};
static std::mutex    g_statusMutex;
static OverlayStatus g_status;

// =====================================================================
//  CAPTURE ENGINE
// =====================================================================
enum class CaptureStatus { Success, Timeout, AccessLost, Error, InvalidROI };

class DXGIFrameLocker {
    IDXGIOutputDuplication* d;
public:
    explicit DXGIFrameLocker(IDXGIOutputDuplication* dup) : d(dup) {}
    ~DXGIFrameLocker() { if (d) d->ReleaseFrame(); }
};

class DXGIMapLocker {
    ID3D11DeviceContext* ctx;
    ID3D11Resource* res;
public:
    DXGIMapLocker(ID3D11DeviceContext* c, ID3D11Resource* r) : ctx(c), res(r) {}
    ~DXGIMapLocker() { if (ctx && res) ctx->Unmap(res, 0); }
};

class DXGICaptureEngine {
    ComPtr<ID3D11Device>           device;
    ComPtr<ID3D11DeviceContext>    context;
    ComPtr<IDXGIOutputDuplication> deskDupl;
    ComPtr<ID3D11Texture2D>        stagingTex;
    int texW = 0, texH = 0;
    DXGI_FORMAT texFormat = DXGI_FORMAT_UNKNOWN;
    bool initialized = false;
    bool isRGBA = false;
public:
    bool IsInitialized() const { return initialized; }
    bool IsRGBA() const        { return isRGBA; }

    bool Init() {
        Cleanup();
        D3D_FEATURE_LEVEL fl;
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                       nullptr, 0, D3D11_SDK_VERSION,
                                       device.GetAddressOf(), &fl, context.GetAddressOf());
        if (FAILED(hr)) return false;

        ComPtr<IDXGIDevice> dxgiDevice;
        hr = device.As(&dxgiDevice); if (FAILED(hr)) return false;
        ComPtr<IDXGIAdapter> dxgiAdapter;
        hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)dxgiAdapter.GetAddressOf());
        if (FAILED(hr)) return false;
        ComPtr<IDXGIOutput> dxgiOutput;
        hr = dxgiAdapter->EnumOutputs(0, dxgiOutput.GetAddressOf());
        if (FAILED(hr)) return false;
        ComPtr<IDXGIOutput1> dxgiOutput1;
        hr = dxgiOutput.As(&dxgiOutput1); if (FAILED(hr)) return false;

        hr = dxgiOutput1->DuplicateOutput(device.Get(), deskDupl.GetAddressOf());
        if (SUCCEEDED(hr)) { initialized = true; return true; }
        return false;
    }

    CaptureStatus GrabFrame(cv::Mat& outFrame, int currentRadius) {
        if (!initialized || !deskDupl) return CaptureStatus::Error;
        outFrame.release();

        DXGI_OUTDUPL_FRAME_INFO info = {};
        ComPtr<IDXGIResource> resource;
        HRESULT hr = deskDupl->AcquireNextFrame(10, &info, resource.GetAddressOf());
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return CaptureStatus::Timeout;
        if (hr == DXGI_ERROR_ACCESS_LOST)  return CaptureStatus::AccessLost;
        if (FAILED(hr)) return CaptureStatus::Error;

        DXGIFrameLocker frameLock(deskDupl.Get());

        ComPtr<ID3D11Texture2D> acquired;
        hr = resource.As(&acquired);
        if (FAILED(hr)) return CaptureStatus::Error;

        D3D11_TEXTURE2D_DESC desc;
        acquired->GetDesc(&desc);

        if      (desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM) isRGBA = true;
        else if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM) isRGBA = false;
        else return CaptureStatus::Error;

        if (!stagingTex || texW != (int)desc.Width || texH != (int)desc.Height || texFormat != desc.Format) {
            stagingTex.Reset();
            desc.Usage          = D3D11_USAGE_STAGING;
            desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            desc.BindFlags      = 0;
            desc.MiscFlags      = 0;
            hr = device->CreateTexture2D(&desc, nullptr, stagingTex.GetAddressOf());
            if (FAILED(hr)) return CaptureStatus::Error;
            texW = desc.Width; texH = desc.Height; texFormat = desc.Format;
        }

        context->CopyResource(stagingTex.Get(), acquired.Get());

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        hr = context->Map(stagingTex.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) return CaptureStatus::Error;

        DXGIMapLocker mapLock(context.Get(), stagingTex.Get());
        if (!mapped.pData) return CaptureStatus::Error;

        int cx = desc.Width / 2;
        int cy = desc.Height / 2;
        int fov = currentRadius + 20;
        cv::Rect roi(cx - fov, cy - fov, fov * 2, fov * 2);

        if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > (int)desc.Width || roi.y + roi.height > (int)desc.Height)
            return CaptureStatus::InvalidROI;

        cv::Mat full(desc.Height, desc.Width, CV_8UC4, mapped.pData, mapped.RowPitch);
        outFrame = full(roi).clone();
        return CaptureStatus::Success;
    }

    void Cleanup() {
        initialized = false;
        stagingTex.Reset(); deskDupl.Reset(); context.Reset(); device.Reset();
        texW = texH = 0; texFormat = DXGI_FORMAT_UNKNOWN;
    }
};

// =====================================================================
//  FIRE CONTROL
// =====================================================================
static void SetFireState(bool fire) {
    if (fire && !g_isFiring.load()) {
        INPUT c = {}; c.type = INPUT_MOUSE; c.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &c, sizeof(INPUT));
        g_isFiring.store(true);
    } else if (!fire && g_isFiring.load()) {
        INPUT c = {}; c.type = INPUT_MOUSE; c.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &c, sizeof(INPUT));
        g_isFiring.store(false);
    }
}

// =====================================================================
//  OVERLAY
// =====================================================================
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM w, LPARAM l) {
    if (msg == WM_ERASEBKGND) return 1;

    if (msg == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        RECT client; GetClientRect(hwnd, &client);

        HBRUSH bg = CreateSolidBrush(RGB(0, 0, 0));
        FillRect(hdc, &client, bg);
        DeleteObject(bg);

        OverlayStatus s;
        { std::lock_guard<std::mutex> lk(g_statusMutex); s = g_status; }

        int cx = GetSystemMetrics(SM_CXSCREEN) / 2;
        int cy = GetSystemMetrics(SM_CYSCREEN) / 2;
        int r  = s.radius;

        COLORREF color = s.firing ? RGB(255, 60, 60)
                       : s.red    ? RGB(255, 220, 0)
                                  : RGB(0, 255, 0);
        HPEN pen = CreatePen(PS_SOLID, 2, color);
        HGDIOBJ oldPen = SelectObject(hdc, pen);
        HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Ellipse(hdc, cx - r, cy - r, cx + r, cy + r);
        SelectObject(hdc, oldPen);
        SelectObject(hdc, oldBrush);
        DeleteObject(pen);

        char buf[256];
        snprintf(buf, sizeof(buf),
                 "FPS %.0f | R %d | LIFT %d | RED:%c ALT:%c FIRE:%c",
                 s.fps, s.radius, s.lift,
                 s.red    ? 'Y' : 'N',
                 s.alt    ? 'Y' : 'N',
                 s.firing ? 'Y' : 'N');
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, color);
        TextOutA(hdc, 20, 20, buf, (int)strlen(buf));

        EndPaint(hwnd, &ps);
        return 0;
    }

    if (msg == WM_TIMER) {
        InvalidateRect(hwnd, nullptr, FALSE);
        return 0;
    }

    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hwnd, msg, w, l);
}

void OverlayThread() {
    g_overlayThreadId.store(GetCurrentThreadId());

    WNDCLASS wc = {};
    wc.lpfnWndProc   = OverlayProc;
    wc.hInstance     = GetModuleHandle(nullptr);
    wc.lpszClassName = "ModarkOverlay";
    if (!RegisterClass(&wc)) { g_overlayState.store(-1); return; }

    HWND hwnd = CreateWindowEx(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "ModarkOverlay", "Overlay", WS_POPUP,
        0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN),
        nullptr, nullptr, wc.hInstance, nullptr);

    if (!hwnd) { UnregisterClass("ModarkOverlay", wc.hInstance); g_overlayState.store(-1); return; }

    g_overlayHwnd.store(hwnd);
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW);
    SetTimer(hwnd, 1, 100, nullptr);
    g_overlayState.store(1);

    MSG m;
    while (GetMessage(&m, nullptr, 0, 0) > 0) { TranslateMessage(&m); DispatchMessage(&m); }

    if (IsWindow(hwnd)) DestroyWindow(hwnd);
    UnregisterClass("ModarkOverlay", wc.hInstance);
}

// =====================================================================
//  HOTKEYS (non-blocking)
// =====================================================================
class Hotkeys {
    std::map<int, Clock::time_point> lastFire;
public:
    bool Once(int vk) {
        static std::map<int, bool> prev;
        bool now = (GetAsyncKeyState(vk) & 0x8000) != 0;
        bool was = prev[vk];
        prev[vk] = now;
        return now && !was;
    }
    bool Repeat(int vk, int cooldownMs) {
        if (!(GetAsyncKeyState(vk) & 0x8000)) return false;
        auto now = Clock::now();
        auto it = lastFire.find(vk);
        if (it != lastFire.end() &&
            duration_cast<milliseconds>(now - it->second).count() < cooldownMs) return false;
        lastFire[vk] = now;
        return true;
    }
};

// =====================================================================
//  CONSOLE HANDLER
// =====================================================================
BOOL WINAPI ConsoleHandler(DWORD sig) {
    if (sig == CTRL_C_EVENT || sig == CTRL_CLOSE_EVENT) {
        g_appRunning.store(false);
        SetFireState(false);
        HWND h = g_overlayHwnd.load();
        if (h) PostMessage(h, WM_CLOSE, 0, 0);
        else {
            DWORD tid = g_overlayThreadId.load();
            if (tid) PostThreadMessage(tid, WM_QUIT, 0, 0);
        }
        return TRUE;
    }
    return FALSE;
}

// =====================================================================
//  MAIN
// =====================================================================
int main() {
    SetProcessDPIAware();
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    std::cout << "[+] Modark Triggerbot v62 — red-only + auto-lift\n";
    std::cout << "[+] Hotkeys: ALT hold = fire | UP/DOWN = lift +/- | +/- = radius | F8 = exit\n";

    Config cfg = LoadConfig("aimbot.ini");
    g_circleRadius.store(cfg.circleRadius);
    g_liftValue.store(cfg.liftValue);
    g_liftIntervalMs.store(cfg.liftIntervalMs);
    g_minRedArea.store(cfg.minRedArea);

    std::cout << "[+] radius=" << cfg.circleRadius
              << " lift=" << cfg.liftValue
              << " liftInterval=" << cfg.liftIntervalMs << "ms"
              << " grace=" << cfg.fireGraceMs << "ms"
              << " minRedArea=" << cfg.minRedArea << "\n";

    std::thread overlay(OverlayThread);
    while (g_overlayState.load() == 0) std::this_thread::sleep_for(milliseconds(10));
    if (g_overlayState.load() == -1) {
        std::cerr << "[-] FATAL: overlay init failed.\n";
        if (overlay.joinable()) overlay.join();
        return 1;
    }

    DXGICaptureEngine capture;
    while (g_appRunning.load() && !capture.Init())
        std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!capture.IsInitialized()) {
        g_appRunning.store(false);
        if (overlay.joinable()) overlay.join();
        return 1;
    }

    std::cout << "[+] Engine live. Hold ALT + red in circle to fire.\n";

    Hotkeys hk;
    auto lastInputCheck = Clock::now();
    auto lastFpsTick    = Clock::now();
    auto lastRedSeen    = Clock::now() - milliseconds(1000);
    auto lastLiftTime   = Clock::now();
    int  frameCount     = 0;

    cv::Mat roi, bgr, hsv, circleMask;
    std::vector<std::vector<cv::Point>> redContours;
    int lastMaskRadius = -1;

    while (g_appRunning.load()) {
        auto now = Clock::now();

        // ---------------- input ----------------
        if (duration_cast<milliseconds>(now - lastInputCheck).count() >= 15) {
            bool alt = (GetAsyncKeyState(TRIGGER_KEY) & 0x8000) != 0;
            g_isAltPressed.store(alt);

            if (hk.Repeat(VK_UP, 60)) {
                g_liftValue.store(std::min(g_liftValue.load() + 1, 50));
                std::cout << "[CTL] liftValue=" << g_liftValue.load() << "\n";
            }
            if (hk.Repeat(VK_DOWN, 60)) {
                g_liftValue.store(std::max(g_liftValue.load() - 1, 0));
                std::cout << "[CTL] liftValue=" << g_liftValue.load() << "\n";
            }

            int r = g_circleRadius.load();
            if (hk.Repeat(VK_ADD, 50) || hk.Repeat(VK_OEM_PLUS, 50)) {
                g_circleRadius.store(std::min(r + 3, 500));
                HWND h = g_overlayHwnd.load(); if (h) InvalidateRect(h, nullptr, FALSE);
            }
            if (hk.Repeat(VK_SUBTRACT, 50) || hk.Repeat(VK_OEM_MINUS, 50)) {
                g_circleRadius.store(std::max(r - 3, 30));
                HWND h = g_overlayHwnd.load(); if (h) InvalidateRect(h, nullptr, FALSE);
            }

            if (hk.Once(VK_F8)) {
                std::cout << "[+] F8 -> shutdown\n";
                g_appRunning.store(false);
                break;
            }

            lastInputCheck = now;
        }

        // ---------------- capture ----------------
        int radius = g_circleRadius.load();
        CaptureStatus st = capture.GrabFrame(roi, radius);
        if (st == CaptureStatus::AccessLost) {
            capture.Cleanup();
            std::this_thread::sleep_for(milliseconds(500));
            while (g_appRunning.load() && !capture.Init())
                std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        } else if (st == CaptureStatus::Timeout) {
            continue;
        } else if (st != CaptureStatus::Success) {
            std::this_thread::sleep_for(milliseconds(2));
            continue;
        }

        if (roi.empty()) continue;

        if (capture.IsRGBA()) cv::cvtColor(roi, bgr, cv::COLOR_RGBA2BGR);
        else                  cv::cvtColor(roi, bgr, cv::COLOR_BGRA2BGR);
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

        int cx = roi.cols / 2, cy = roi.rows / 2;

        if (radius != lastMaskRadius) {
            circleMask = cv::Mat::zeros(roi.size(), CV_8UC1);
            cv::circle(circleMask, cv::Point(cx, cy), radius, cv::Scalar(255), -1);
            lastMaskRadius = radius;
        }

        // ---------------- red detection ----------------
        cv::Mat m1, m2, redMask, finalRed;
        cv::inRange(hsv, cv::Scalar(0, 120, 100),   cv::Scalar(10, 255, 255), m1);
        cv::inRange(hsv, cv::Scalar(170, 120, 100), cv::Scalar(180, 255, 255), m2);
        cv::bitwise_or(m1, m2, redMask);
        cv::bitwise_and(redMask, circleMask, finalRed);

        redContours.clear();
        cv::findContours(finalRed, redContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        bool redDetected = false;
        int minArea = g_minRedArea.load();
        for (auto& c : redContours) {
            if (cv::contourArea(c) >= minArea) { redDetected = true; break; }
        }
        g_redVisible.store(redDetected);
        if (redDetected) lastRedSeen = now;

        // ---------------- fire logic ----------------
        long sinceRed = duration_cast<milliseconds>(now - lastRedSeen).count();
        bool shouldFire = g_isAltPressed.load() && (redDetected || sinceRed < cfg.fireGraceMs);

        if (shouldFire) {
            bool wasFiring = g_isFiring.load();
            SetFireState(true);

            if (!wasFiring) {
                // reset lift tick on fresh engagement -> first lift immediate
                lastLiftTime = now - milliseconds(g_liftIntervalMs.load());
            }

            if (duration_cast<milliseconds>(now - lastLiftTime).count() >= g_liftIntervalMs.load()) {
                int lift = g_liftValue.load();
                if (lift > 0) {
                    INPUT li = {};
                    li.type = INPUT_MOUSE;
                    li.mi.dy = -lift;              // negative Y = up
                    li.mi.dwFlags = MOUSEEVENTF_MOVE;
                    SendInput(1, &li, sizeof(INPUT));
                }
                lastLiftTime = now;
            }
        } else {
            SetFireState(false);
            lastLiftTime = now;
        }

        // ---------------- publish status ----------------
        {
            std::lock_guard<std::mutex> lk(g_statusMutex);
            g_status.red     = redDetected;
            g_status.alt     = g_isAltPressed.load();
            g_status.firing  = g_isFiring.load();
            g_status.radius  = radius;
            g_status.lift    = g_liftValue.load();
            g_status.fps     = g_fps.load();
        }

        // ---------------- fps ----------------
        frameCount++;
        long sinceFps = duration_cast<milliseconds>(now - lastFpsTick).count();
        if (sinceFps >= 500) {
            g_fps.store(frameCount * 1000.0f / sinceFps);
            frameCount = 0;
            lastFpsTick = now;
        }

        std::this_thread::sleep_for(milliseconds(1));
    }

    // ---------------- shutdown ----------------
    g_appRunning.store(false);
    SetFireState(false);
    HWND h = g_overlayHwnd.load();
    if (h) PostMessage(h, WM_CLOSE, 0, 0);
    else {
        DWORD tid = g_overlayThreadId.load();
        if (tid) PostThreadMessage(tid, WM_QUIT, 0, 0);
    }
    if (overlay.joinable()) overlay.join();
    capture.Cleanup();
    std::cout << "[+] Clean exit.\n";
    return 0;
}
