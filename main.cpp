// language: C++, file: main.cpp, runtime: MSVC v143 (C++17), Windows 10/11
// v79 - YOLOv8 AI Integration + Advanced Debug Logging

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
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
#include <random>
#include <functional>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

// =====================================================================
//  LOGGING
// =====================================================================
static void LogInfo(const std::string& msg) {
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_INTENSITY);
    std::cout << "[INFO] " << msg << "\n";
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
}
static void LogError(const std::string& msg, HRESULT hr = S_OK) {
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_INTENSITY);
    std::cout << "[ERROR] " << msg;
    if (hr != S_OK) std::cout << " (HRESULT: 0x" << std::hex << hr << std::dec << ")";
    std::cout << "\n";
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
}

// =====================================================================
//  CONFIG & GLOBALS
// =====================================================================
struct Config {
    int circleRadius   = 150;
    float aimSpeed     = 0.65f;
    int headOffset     = 0;
    int liftValue      = 3;
    int liftIntervalMs = 30;
    float confThreshold= 0.40f; // AI Confidence threshold (0.0 to 1.0)
    float nmsThreshold = 0.45f;
};

static Config LoadConfig(const char* path) {
    Config c; std::ifstream f(path);
    if (!f) return c;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        auto eq = line.find('='); if (eq == std::string::npos) continue;
        std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        while (!k.empty() && (k.back() == ' ' || k.back() == '\t')) k.pop_back();
        while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(0, 1);
        try {
            if      (k == "circleRadius")   c.circleRadius   = std::stoi(v);
            else if (k == "aimSpeed")       c.aimSpeed       = std::stof(v);
            else if (k == "headOffset")     c.headOffset     = std::stoi(v);
            else if (k == "liftValue")      c.liftValue      = std::stoi(v);
            else if (k == "liftIntervalMs") c.liftIntervalMs = std::stoi(v);
            else if (k == "confThreshold")  c.confThreshold  = std::stof(v);
            else if (k == "nmsThreshold")   c.nmsThreshold   = std::stof(v);
        } catch (...) {}
    }
    return c;
}

static std::atomic<bool>  g_appRunning(true);
static std::atomic<int>   g_circleRadius(150);
static std::atomic<float> g_aimSpeed(0.65f);
static std::atomic<int>   g_headOffset(0);
static std::atomic<int>   g_liftValue(3);
static std::atomic<int>   g_liftIntervalMs(30);
static std::atomic<float> g_confThreshold(0.40f);
static std::atomic<float> g_nmsThreshold(0.45f);

static std::atomic<bool>  g_isFiring(false);
static std::atomic<bool>  g_isAltPressed(false);
static std::atomic<bool>  g_targetVisible(false);
static std::atomic<DWORD> g_overlayThreadId(0);
static std::atomic<HWND>  g_overlayHwnd(nullptr);
static std::atomic<int>   g_overlayState(0);
static std::atomic<float> g_fps(0.0f);

static const int TRIGGER_KEY = VK_MENU;

struct OverlayStatus {
    bool  red = false, alt = false, firing = false;
    int   radius = 150, lift = 3;
    float aimSpd = 0.65f, fps = 0.0f;
    int   boxX = 0, boxY = 0, boxW = 0, boxH = 0;
    float confidence = 0.0f;
};
static std::mutex    g_statusMutex;
static OverlayStatus g_status;

// =====================================================================
//  AI ENGINE (YOLOv8)
// =====================================================================
struct AIDetection {
    cv::Rect box;
    float confidence;
};

class AIEngine {
    cv::dnn::Net net;
    bool isLoaded = false;
    int inputWidth = 640;
    int inputHeight = 640;
public:
    bool LoadModel(const std::string& modelPath) {
        try {
            net = cv::dnn::readNetFromONNX(modelPath);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            isLoaded = true;
            LogInfo("AI Model Loaded Successfully: " + modelPath);
            return true;
        } catch (const std::exception& e) {
            LogError("Failed to load AI model: " + std::string(e.what()));
            isLoaded = false;
            return false;
        }
    }

    bool IsLoaded() const { return isLoaded; }

    std::vector<AIDetection> Detect(cv::Mat& frame, float confThresh, float nmsThresh) {
        std::vector<AIDetection> detections;
        if (!isLoaded) return detections;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat outputs = net.forward();

        cv::Mat outputMat(outputs.size[1], outputs.size[2], CV_32F, outputs.ptr<float>());
        cv::transpose(outputMat, outputMat);

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        float xFactor = (float)frame.cols / inputWidth;
        float yFactor = (float)frame.rows / inputHeight;

        for (int i = 0; i < outputMat.rows; ++i) {
            float* row = outputMat.ptr<float>(i);
            
            // For 1 class (enemy), row[4] is the confidence
            float confidence = row[4]; 
            if (confidence < confThresh) continue;

            float cx = row[0];
            float cy = row[1];
            float w = row[2];
            float h = row[3];

            int left = int((cx - 0.5 * w) * xFactor);
            int top = int((cy - 0.5 * h) * yFactor);
            int width = int(w * xFactor);
            int height = int(h * yFactor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
            classIds.push_back(0);
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);

        for (int idx : indices) {
            detections.push_back({boxes[idx], confidences[idx]});
        }

        return detections;
    }
};

// =====================================================================
//  CAPTURE ENGINE
// =====================================================================
enum class CaptureStatus { Success, Timeout, AccessLost, DeviceLost, Error, InvalidROI };

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
        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, device.GetAddressOf(), &fl, context.GetAddressOf());
        if (FAILED(hr)) { LogError("D3D11CreateDevice failed", hr); return false; }

        ComPtr<IDXGIDevice> dxgiDevice; hr = device.As(&dxgiDevice); if (FAILED(hr)) return false;
        ComPtr<IDXGIAdapter> dxgiAdapter; hr = dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)dxgiAdapter.GetAddressOf()); if (FAILED(hr)) return false;
        ComPtr<IDXGIOutput> dxgiOutput; hr = dxgiAdapter->EnumOutputs(0, dxgiOutput.GetAddressOf()); if (FAILED(hr)) return false;
        ComPtr<IDXGIOutput1> dxgiOutput1; hr = dxgiOutput.As(&dxgiOutput1); if (FAILED(hr)) return false;

        hr = dxgiOutput1->DuplicateOutput(device.Get(), deskDupl.GetAddressOf());
        if (SUCCEEDED(hr)) { initialized = true; return true; }
        LogError("DuplicateOutput failed", hr);
        return false;
    }

    CaptureStatus GrabFrameAndProcess(int currentRadius, std::function<void(cv::Mat&, bool, int, int)> processor) {
        if (!initialized || !deskDupl) return CaptureStatus::Error;

        DXGI_OUTDUPL_FRAME_INFO info = {};
        ComPtr<IDXGIResource> resource;
        HRESULT hr = deskDupl->AcquireNextFrame(10, &info, resource.GetAddressOf());
        
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return CaptureStatus::Timeout;
        if (hr == DXGI_ERROR_ACCESS_LOST)  return CaptureStatus::AccessLost;
        if (hr == DXGI_ERROR_DEVICE_REMOVED || hr == DXGI_ERROR_DEVICE_RESET) return CaptureStatus::DeviceLost;
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

        int cx = desc.Width / 2;
        int cy = desc.Height / 2;
        int fov = currentRadius + 20;
        int cropW = fov * 2, cropH = fov * 2;
        int cropX = std::max(0, cx - fov);
        int cropY = std::max(0, cy - fov);

        if (cropX + cropW > (int)desc.Width) cropW = desc.Width - cropX;
        if (cropY + cropH > (int)desc.Height) cropH = desc.Height - cropY;
        if (cropW <= 0 || cropH <= 0) return CaptureStatus::InvalidROI;

        if (!stagingTex || texW != cropW || texH != cropH || texFormat != desc.Format) {
            stagingTex.Reset();
            D3D11_TEXTURE2D_DESC sdesc = {};
            sdesc.Width = cropW; sdesc.Height = cropH; sdesc.MipLevels = 1; sdesc.ArraySize = 1;
            sdesc.Format = desc.Format; sdesc.SampleDesc.Count = 1;
            sdesc.Usage = D3D11_USAGE_STAGING; sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            hr = device->CreateTexture2D(&sdesc, nullptr, stagingTex.GetAddressOf());
            if (FAILED(hr)) return CaptureStatus::Error;
            texW = cropW; texH = cropH; texFormat = desc.Format;
        }

        D3D11_BOX box;
        box.left = cropX; box.top = cropY; box.front = 0;
        box.right = cropX + cropW; box.bottom = cropY + cropH; box.back = 1;
        context->CopySubresourceRegion(stagingTex.Get(), 0, 0, 0, 0, acquired.Get(), 0, &box);
        
        if (device->GetDeviceRemovedReason() != S_OK) return CaptureStatus::DeviceLost;

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        hr = context->Map(stagingTex.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) return CaptureStatus::Error;

        {
            DXGIMapLocker mapLock(context.Get(), stagingTex.Get());
            if (!mapped.pData) return CaptureStatus::Error;

            int realCx = cx - cropX;
            int realCy = cy - cropY;

            cv::Mat mappedMat(cropH, cropW, CV_8UC4, mapped.pData, mapped.RowPitch);
            processor(mappedMat, isRGBA, realCx, realCy);
        }

        return CaptureStatus::Success;
    }

    void Cleanup() {
        initialized = false;
        stagingTex.Reset(); deskDupl.Reset(); context.Reset(); device.Reset();
        texW = texH = 0; texFormat = DXGI_FORMAT_UNKNOWN;
    }
};

// =====================================================================
//  FIRE CONTROL & OVERLAY & HOTKEYS
// =====================================================================
static void SetFireState(bool fire) {
    if (fire && !g_isFiring.load()) {
        INPUT c = {}; c.type = INPUT_MOUSE; c.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
        SendInput(1, &c, sizeof(INPUT)); g_isFiring.store(true);
    } else if (!fire && g_isFiring.load()) {
        INPUT c = {}; c.type = INPUT_MOUSE; c.mi.dwFlags = MOUSEEVENTF_LEFTUP;
        SendInput(1, &c, sizeof(INPUT)); g_isFiring.store(false);
    }
}

LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM w, LPARAM l) {
    if (msg == WM_ERASEBKGND) return 1;
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps; HDC hdc = BeginPaint(hwnd, &ps);
        RECT client; GetClientRect(hwnd, &client);
        HBRUSH bg = CreateSolidBrush(RGB(0, 0, 0)); FillRect(hdc, &client, bg); DeleteObject(bg);

        OverlayStatus s; { std::lock_guard<std::mutex> lk(g_statusMutex); s = g_status; }
        int cx = GetSystemMetrics(SM_CXSCREEN) / 2, cy = GetSystemMetrics(SM_CYSCREEN) / 2, r = s.radius;

        COLORREF color = s.firing ? RGB(255, 60, 60) : s.red ? RGB(255, 220, 0) : RGB(0, 255, 0);
        HPEN pen = CreatePen(PS_SOLID, 2, color);
        HGDIOBJ oldPen = SelectObject(hdc, pen);
        HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Ellipse(hdc, cx - r, cy - r, cx + r, cy + r);
        SelectObject(hdc, oldPen); SelectObject(hdc, oldBrush); DeleteObject(pen);

        if (s.red && s.boxW > 0 && s.boxH > 0) {
            int fov = r + 20;
            int screenStartX = cx - fov;
            int screenStartY = cy - fov;
            int drawX = screenStartX + s.boxX;
            int drawY = screenStartY + s.boxY;

            HPEN boxPen = CreatePen(PS_SOLID, 2, RGB(255, 220, 0));
            HGDIOBJ oldBoxPen = SelectObject(hdc, boxPen);
            HGDIOBJ oldBoxBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
            Rectangle(hdc, drawX, drawY, drawX + s.boxW, drawY + s.boxH);
            SelectObject(hdc, oldBoxPen); SelectObject(hdc, oldBoxBrush); DeleteObject(boxPen);
        }

        char buf[256];
        snprintf(buf, sizeof(buf), "FPS %.0f | R %d | LIFT %d | SPD %.2f | AI:%.0f%% | ALT:%c | FIRE:%c",
                 s.fps, s.radius, s.lift, s.aimSpd, s.confidence * 100, s.alt ? 'Y' : 'N', s.firing ? 'Y' : 'N');
        SetBkMode(hdc, TRANSPARENT); SetTextColor(hdc, color);
        TextOutA(hdc, 20, 20, buf, (int)strlen(buf));
        EndPaint(hwnd, &ps); return 0;
    }
    if (msg == WM_TIMER) { InvalidateRect(hwnd, nullptr, FALSE); return 0; }
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hwnd, msg, w, l);
}

void OverlayThread() {
    g_overlayThreadId.store(GetCurrentThreadId());
    WNDCLASS wc = {}; wc.lpfnWndProc = OverlayProc; wc.hInstance = GetModuleHandle(nullptr); wc.lpszClassName = "ModarkOverlay";
    if (!RegisterClass(&wc)) {
        DWORD err = GetLastError();
        if (err != ERROR_CLASS_ALREADY_EXISTS) { g_overlayState.store(-1); return; }
    }
    HWND hwnd = CreateWindowEx(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "ModarkOverlay", "Overlay", WS_POPUP, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), nullptr, nullptr, wc.hInstance, nullptr);
    if (!hwnd) { g_overlayState.store(-1); return; }
    g_overlayHwnd.store(hwnd);
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW); SetTimer(hwnd, 1, 100, nullptr); g_overlayState.store(1);
    MSG m; while (GetMessage(&m, nullptr, 0, 0) > 0) { TranslateMessage(&m); DispatchMessage(&m); }
    if (IsWindow(hwnd)) DestroyWindow(hwnd); UnregisterClass("ModarkOverlay", wc.hInstance);
}

class Hotkeys {
    std::map<int, Clock::time_point> lastFire;
public:
    bool Once(int vk) { static std::map<int, bool> prev; bool now = (GetAsyncKeyState(vk) & 0x8000) != 0; bool was = prev[vk]; prev[vk] = now; return now && !was; }
    bool Repeat(int vk, int cooldownMs) {
        if (!(GetAsyncKeyState(vk) & 0x8000)) return false;
        auto now = Clock::now(); auto it = lastFire.find(vk);
        if (it != lastFire.end() && duration_cast<milliseconds>(now - it->second).count() < cooldownMs) return false;
        lastFire[vk] = now; return true;
    }
};

BOOL WINAPI ConsoleHandler(DWORD sig) {
    if (sig == CTRL_C_EVENT || sig == CTRL_CLOSE_EVENT) {
        g_appRunning.store(false); SetFireState(false);
        HWND h = g_overlayHwnd.load(); if (h) PostMessage(h, WM_CLOSE, 0, 0);
        else { DWORD tid = g_overlayThreadId.load(); if (tid) PostThreadMessage(tid, WM_QUIT, 0, 0); }
        return TRUE;
    }
    return FALSE;
}

// =====================================================================
//  MAIN
// =====================================================================
int main() {
    SetProcessDPIAware(); SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    LogInfo("Modark Aimbot v79 - YOLOv8 Full Release");

    Config cfg = LoadConfig("aimbot.ini");
    g_circleRadius.store(cfg.circleRadius); g_aimSpeed.store(cfg.aimSpeed);
    g_headOffset.store(cfg.headOffset); g_liftValue.store(cfg.liftValue);
    g_liftIntervalMs.store(cfg.liftIntervalMs); g_confThreshold.store(cfg.confThreshold);
    g_nmsThreshold.store(cfg.nmsThreshold);

    AIEngine ai;
    // تأكد أن ملف best.onnx في نفس المجلد
    if (!ai.LoadModel("best.onnx")) {
        LogError("FATAL: Could not load 'best.onnx'. Place it next to the exe.");
        std::this_thread::sleep_for(std::chrono::seconds(5));
        return 1;
    }

    std::thread overlay(OverlayThread);
    while (g_overlayState.load() == 0) std::this_thread::sleep_for(milliseconds(10));
    if (g_overlayState.load() == -1) { LogError("Overlay init failed."); return 1; }

    DXGICaptureEngine capture;
    while (g_appRunning.load() && !capture.Init()) { LogInfo("Retrying D3D11..."); std::this_thread::sleep_for(std::chrono::seconds(1)); }
    if (!capture.IsInitialized()) { g_appRunning.store(false); if (overlay.joinable()) overlay.join(); return 1; }

    Hotkeys hk;
    auto lastInputCheck = Clock::now(), lastFpsTick = Clock::now(), lastLogTick = Clock::now();
    auto lastTargetSeen = Clock::now() - milliseconds(1000), lastLiftTime = Clock::now();
    int frameCount = 0;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> delayDist(20, 50);
    std::uniform_int_distribution<int> jitterDist(-1, 1); 
    
    auto targetAcquiredTime = Clock::now();
    bool wasTargetDetected = false; int currentHumanDelay = 30;

    cv::Mat bgr, circleMask;
    int lastMaskRadius = -1;
    float accumX = 0.0f, accumY = 0.0f;

    auto processor = [&](cv::Mat& mappedMat, bool isRGBA, int realCx, int realCy) {
        
        if (isRGBA) cv::cvtColor(mappedMat, bgr, cv::COLOR_RGBA2BGR);
        else        cv::cvtColor(mappedMat, bgr, cv::COLOR_BGRA2BGR);

        int radius = g_circleRadius.load();

        if (radius != lastMaskRadius || circleMask.size() != mappedMat.size()) {
            circleMask = cv::Mat::zeros(mappedMat.size(), CV_8UC1);
            cv::circle(circleMask, cv::Point(realCx, realCy), radius, cv::Scalar(255), -1);
            lastMaskRadius = radius;
        }

        // AI Detection on masked image
        std::vector<AIDetection> detections = ai.Detect(bgr, g_confThreshold.load(), g_nmsThreshold.load());

        bool targetFound = false;
        cv::Rect bestRect;
        float bestConf = 0.0f;
        double minDist = 999999;

        for (auto& det : detections) {
            // Only accept boxes inside the circle
            cv::Point center(det.box.x + det.box.width/2, det.box.y + det.box.height/2);
            double distToCenter = std::hypot(center.x - realCx, center.y - realCy);
            
            if (distToCenter > radius) continue; // Outside green circle

            if (distToCenter < minDist) { 
                minDist = distToCenter; 
                bestRect = det.box; 
                bestConf = det.confidence;
                targetFound = true; 
            }
        }

        g_targetVisible.store(targetFound);
        auto now = Clock::now();
        if (targetFound) lastTargetSeen = now;

        // =================================================================
        // EXTENSIVE DEBUG LOGGING (Prints every 500ms)
        // =================================================================
        if (duration_cast<milliseconds>(now - lastLogTick).count() > 500) {
            if (targetFound) {
                std::cout << "[DEBUG] AI Target Locked! Conf: " << (int)(bestConf*100) << "% | ";
            } else {
                std::cout << "[DEBUG] Searching... No AI Target in circle. | ";
            }
            
            std::cout << "ALT: " << (g_isAltPressed.load() ? "ON" : "OFF");
            
            if (!targetFound) std::cout << " -> (Waiting for target)";
            else if (!g_isAltPressed.load()) std::cout << " -> (Holding fire, press ALT)";
            else if (!g_isFiring.load()) std::cout << " -> (Waiting for human delay)";
            else std::cout << " -> (FIRING)";
            
            std::cout << "\n";
            lastLogTick = now;
        }
        // =================================================================

        auto sinceTarget = duration_cast<milliseconds>(now - lastTargetSeen).count();
        
        // FIRE ONLY IF ALT IS HELD AND TARGET FOUND
        bool shouldFire = g_isAltPressed.load() && (targetFound || sinceTarget < 50);

        if (targetFound) {
            if (!wasTargetDetected) { targetAcquiredTime = now; wasTargetDetected = true; currentHumanDelay = delayDist(rng); }
        } else wasTargetDetected = false;

        if (shouldFire && targetFound) {
            
            int jitterX = jitterDist(rng);
            int jitterY = jitterDist(rng);
            
            // Aim at 15% from the top (head area)
            int headX = bestRect.x + bestRect.width / 2 + jitterX;
            int headY = bestRect.y + (int)(bestRect.height * 0.15f) + g_headOffset.load() + jitterY;
            
            float dx = (float)(headX - realCx) * g_aimSpeed.load();
            float dy = (float)(headY - realCy) * g_aimSpeed.load();

            accumX += dx;
            accumY += dy;

            int moveX = (int)accumX;
            int moveY = (int)accumY;

            if (moveX != 0 || moveY != 0) {
                accumX -= moveX;
                accumY -= moveY;

                int steps = 2 + (rng() % 2);
                int stepX = moveX / steps;
                int stepY = moveY / steps;

                for (int i = 0; i < steps; i++) {
                    INPUT in = {}; in.type = INPUT_MOUSE;
                    in.mi.dx = (i == steps - 1) ? (moveX - stepX * (steps - 1)) : stepX;
                    in.mi.dy = (i == steps - 1) ? (moveY - stepY * (steps - 1)) : stepY;
                    in.mi.dwFlags = MOUSEEVENTF_MOVE;
                    SendInput(1, &in, sizeof(INPUT));
                }
            }

            auto timeSinceAcquired = duration_cast<milliseconds>(now - targetAcquiredTime).count();
            if (timeSinceAcquired >= currentHumanDelay) {
                bool wasFiring = g_isFiring.load(); SetFireState(true);
                if (!wasFiring) lastLiftTime = now - milliseconds(g_liftIntervalMs.load());

                if (duration_cast<milliseconds>(now - lastLiftTime).count() >= g_liftIntervalMs.load()) {
                    int baseLift = g_liftValue.load();
                    auto firingDuration = duration_cast<milliseconds>(now - targetAcquiredTime).count();
                    int currentLift = baseLift;
                    
                    if (firingDuration > 300) currentLift = std::max(0, baseLift - 1);
                    if (firingDuration > 800) currentLift = std::max(0, baseLift - 2);

                    if (currentLift > 0) {
                        INPUT li = {}; li.type = INPUT_MOUSE; li.mi.dy = -currentLift; li.mi.dwFlags = MOUSEEVENTF_MOVE;
                        SendInput(1, &li, sizeof(INPUT));
                    }
                    lastLiftTime = now;
                }
            }
        } else {
            SetFireState(false); lastLiftTime = now;
            accumX = 0.0f; accumY = 0.0f;
        }

        frameCount++;
        auto sinceFps = duration_cast<milliseconds>(now - lastFpsTick).count();
        if (sinceFps >= 500) { g_fps.store(frameCount * 1000.0f / sinceFps); frameCount = 0; lastFpsTick = now; }

        {
            std::lock_guard<std::mutex> lk(g_statusMutex);
            g_status.red = targetFound; g_status.alt = g_isAltPressed.load(); g_status.firing = g_isFiring.load();
            g_status.radius = radius; g_status.lift = g_liftValue.load(); g_status.aimSpd = g_aimSpeed.load(); g_status.fps = g_fps.load();
            g_status.confidence = bestConf;
            
            if (targetFound) {
                g_status.boxX = bestRect.x; g_status.boxY = bestRect.y;
                g_status.boxW = bestRect.width; g_status.boxH = bestRect.height;
            } else { g_status.boxW = 0; g_status.boxH = 0; }
        }
    };

    LogInfo("Engine live. AI is looking for enemies. Check console for details.");

    while (g_appRunning.load()) {
        auto now = Clock::now();

        if (duration_cast<milliseconds>(now - lastInputCheck).count() >= 15) {
            g_isAltPressed.store((GetAsyncKeyState(TRIGGER_KEY) & 0x8000) != 0);
            if (hk.Repeat(VK_UP, 60)) g_liftValue.store(std::min(g_liftValue.load() + 1, 50));
            if (hk.Repeat(VK_DOWN, 60)) g_liftValue.store(std::max(g_liftValue.load() - 1, 0));
            if (hk.Repeat(VK_LEFT, 60)) g_aimSpeed.store(std::max(g_aimSpeed.load() - 0.05f, 0.10f));
            if (hk.Repeat(VK_RIGHT, 60)) g_aimSpeed.store(std::min(g_aimSpeed.load() + 0.05f, 1.00f));
            
            if (hk.Repeat(VK_ADD, 50) || hk.Repeat(VK_OEM_PLUS, 50)) { g_circleRadius.store(std::min(g_circleRadius.load() + 3, 500)); HWND h = g_overlayHwnd.load(); if (h) InvalidateRect(h, nullptr, FALSE); }
            if (hk.Repeat(VK_SUBTRACT, 50) || hk.Repeat(VK_OEM_MINUS, 50)) { g_circleRadius.store(std::max(g_circleRadius.load() - 3, 30)); HWND h = g_overlayHwnd.load(); if (h) InvalidateRect(h, nullptr, FALSE); }
            if (hk.Once(VK_F8)) { g_appRunning.store(false); break; }
            
            lastInputCheck = now;
        }

        int radius = g_circleRadius.load();
        CaptureStatus st = capture.GrabFrameAndProcess(radius, processor);
        
        if (st == CaptureStatus::AccessLost) {
            LogError("Capture access lost. Re-initializing...");
            capture.Cleanup(); std::this_thread::sleep_for(milliseconds(500));
            while (g_appRunning.load() && !capture.Init()) std::this_thread::sleep_for(std::chrono::seconds(1));
        } 
        else if (st == CaptureStatus::DeviceLost) {
            LogError("D3D11 Device Lost. Rebuilding...");
            capture.Cleanup(); std::this_thread::sleep_for(std::chrono::milliseconds(2000));
            while (g_appRunning.load() && !capture.Init()) std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        else if (st == CaptureStatus::Success) { std::this_thread::yield(); } 
        else { std::this_thread::sleep_for(milliseconds(2)); }
    }

    g_appRunning.store(false); SetFireState(false);
    HWND h = g_overlayHwnd.load(); if (h) PostMessage(h, WM_CLOSE, 0, 0);
    else { DWORD tid = g_overlayThreadId.load(); if (tid) PostThreadMessage(tid, WM_QUIT, 0, 0); }
    if (overlay.joinable()) overlay.join();
    capture.Cleanup();
    return 0;
}
