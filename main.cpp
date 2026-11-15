// language: C++, file: main.cpp, runtime: MSVC v143 (C++17), Windows 10/11
// v111 - Final Release: Auto-Class Detection, Fixed Overlay, Clean Shutdown

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

using Microsoft::WRL::ComPtr;
using Clock = std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::microseconds;
using std::chrono::duration_cast;

// =====================================================================
//  CONFIGURATION
// =====================================================================
struct Config {
    std::string modelPath = "best.onnx";
    int inputSize = 640;
    int expectedClasses = 1; // Kept for compatibility, but ignored by AI
    float confidenceThreshold = 0.35f;
    float nmsThreshold = 0.45f;
    
    int circleRadius = 150;
    float aimSpeed = 1.0f;
    int headOffsetX = 0;
    int headOffsetY = 0;
    int liftValue = 3;
    int liftIntervalMs = 30;
    float predictionFactor = 1.0f;
    int windMouse = 0;
};

static Config cfg;

static std::string Trim(std::string s) {
    const char* ws = " \t\r\n";
    size_t first = s.find_first_not_of(ws);
    if (first == std::string::npos) return {};
    size_t last = s.find_last_not_of(ws);
    return s.substr(first, last - first + 1);
}

static bool LoadConfig(const std::string& path, Config& c) {
    std::ifstream file(path);
    if (!file) {
        std::cout << "[INFO] config.ini not found. Using defaults.\n";
        return true;
    }

    std::string line;
    try {
        while (std::getline(file, line)) {
            line = Trim(line);
            if (line.empty() || line[0] == '#' || line[0] == ';' || line[0] == '[') continue;
            size_t eq = line.find('=');
            if (eq == std::string::npos) continue;

            std::string key = Trim(line.substr(0, eq));
            std::string value = Trim(line.substr(eq + 1));

            if (key == "model") c.modelPath = value;
            else if (key == "input_size") c.inputSize = std::stoi(value);
            else if (key == "confidence") c.confidenceThreshold = std::stof(value);
            else if (key == "nms") c.nmsThreshold = std::stof(value);
            else if (key == "circleRadius") c.circleRadius = std::stoi(value);
            else if (key == "aimSpeed") c.aimSpeed = std::stof(value);
            else if (key == "headOffsetX") c.headOffsetX = std::stoi(value);
            else if (key == "headOffsetY") c.headOffsetY = std::stoi(value);
            else if (key == "liftValue") c.liftValue = std::stoi(value);
            else if (key == "liftIntervalMs") c.liftIntervalMs = std::stoi(value);
            else if (key == "predictionFactor") c.predictionFactor = std::stof(value);
            else if (key == "windMouse") c.windMouse = std::stoi(value);
        }
    } catch (...) { 
        std::cout << "[-] Error parsing config.ini\n";
        return false; 
    }
    return true;
}

// =====================================================================
//  GLOBALS
// =====================================================================
static std::atomic<bool> g_appRunning(true);
static std::atomic<bool> g_isAltPressed(false);
static std::atomic<bool> g_isFiring(false);
static std::atomic<int>  g_circleRadius(150);
static std::atomic<float> g_aimSpeed(1.0f);
static std::atomic<int>   g_headOffsetX(0);
static std::atomic<int>   g_headOffsetY(0);
static std::atomic<int>   g_liftValue(3);
static std::atomic<int>   g_liftIntervalMs(30);

static std::atomic<HWND> g_overlayHwnd(nullptr);
static std::atomic<int>  g_overlayState(0);

struct TargetData {
    bool found = false;
    int x = 0, y = 0;
    float vx = 0.0f, vy = 0.0f;
    float conf = 0.0f;
    int boxX = 0, boxY = 0, boxW = 0, boxH = 0;
    Clock::time_point detectionTime; 
};
static std::mutex targetMutex;
static TargetData globalTarget;

// =====================================================================
//  DXGI CAPTURE ENGINE
// =====================================================================
enum class GrabStatus { Ok, Timeout, AccessLost, Error };

class DesktopCapture {
public:
    bool Initialize() {
        Cleanup();
        D3D_FEATURE_LEVEL featureLevel{};
        const D3D_FEATURE_LEVEL levels[] = { D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };

        HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels, ARRAYSIZE(levels), D3D11_SDK_VERSION, &device_, &featureLevel, &context_);
        
        if (hr == E_INVALIDARG) {
            const D3D_FEATURE_LEVEL fallback[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1, D3D_FEATURE_LEVEL_10_0 };
            hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, fallback, ARRAYSIZE(fallback), D3D11_SDK_VERSION, &device_, &featureLevel, &context_);
        }

        if (FAILED(hr)) return false;

        ComPtr<IDXGIDevice> dxgiDevice;
        hr = device_.As(&dxgiDevice);
        if (FAILED(hr)) return false;

        ComPtr<IDXGIAdapter> adapter;
        hr = dxgiDevice->GetAdapter(&adapter);
        if (FAILED(hr)) return false;

        ComPtr<IDXGIOutput> output;
        hr = adapter->EnumOutputs(0, &output);
        if (FAILED(hr)) return false;

        ComPtr<IDXGIOutput1> output1;
        hr = output.As(&output1);
        if (FAILED(hr)) return false;

        hr = output1->DuplicateOutput(device_.Get(), &duplication_);
        if (FAILED(hr)) return false;

        return true;
    }

    GrabStatus Grab(cv::Mat& frame) {
        if (!duplication_ || !device_ || !context_) return GrabStatus::Error;

        DXGI_OUTDUPL_FRAME_INFO info{};
        ComPtr<IDXGIResource> resource;
        HRESULT hr = duplication_->AcquireNextFrame(1, &info, &resource);

        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return GrabStatus::Timeout;
        if (hr == DXGI_ERROR_ACCESS_LOST) return GrabStatus::AccessLost;
        if (FAILED(hr)) return GrabStatus::Error;

        ComPtr<ID3D11Texture2D> desktopTexture;
        hr = resource.As(&desktopTexture);
        if (FAILED(hr)) { duplication_->ReleaseFrame(); return GrabStatus::Error; }

        D3D11_TEXTURE2D_DESC desc{};
        desktopTexture->GetDesc(&desc);

        if (!staging_ || desc.Width != stagingWidth_ || desc.Height != stagingHeight_ || desc.Format != stagingFormat_) {
            staging_.Reset();
            D3D11_TEXTURE2D_DESC stagingDesc = desc;
            stagingDesc.BindFlags = 0; stagingDesc.MiscFlags = 0;
            stagingDesc.Usage = D3D11_USAGE_STAGING; stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            hr = device_->CreateTexture2D(&stagingDesc, nullptr, &staging_);
            if (FAILED(hr)) { duplication_->ReleaseFrame(); return GrabStatus::Error; }
            stagingWidth_ = desc.Width; stagingHeight_ = desc.Height; stagingFormat_ = desc.Format;
        }

        context_->CopyResource(staging_.Get(), desktopTexture.Get());
        duplication_->ReleaseFrame();

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hr = context_->Map(staging_.Get(), 0, D3D11_MAP_READ, 0, &mapped);
        if (FAILED(hr)) return GrabStatus::Error;

        cv::Mat raw(static_cast<int>(desc.Height), static_cast<int>(desc.Width), CV_8UC4, mapped.pData, mapped.RowPitch);
        cv::Mat owned = raw.clone();
        context_->Unmap(staging_.Get(), 0);

        if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM)
            cv::cvtColor(owned, frame, cv::COLOR_BGRA2BGR);
        else
            cv::cvtColor(owned, frame, cv::COLOR_RGBA2BGR);

        return GrabStatus::Ok;
    }

    void Cleanup() {
        duplication_.Reset(); staging_.Reset(); context_.Reset(); device_.Reset();
        stagingWidth_ = stagingHeight_ = 0;
    }

private:
    ComPtr<ID3D11Device> device_;
    ComPtr<ID3D11DeviceContext> context_;
    ComPtr<IDXGIOutputDuplication> duplication_;
    ComPtr<ID3D11Texture2D> staging_;
    UINT stagingWidth_ = 0, stagingHeight_ = 0;
    DXGI_FORMAT stagingFormat_ = DXGI_FORMAT_UNKNOWN;
};

// =====================================================================
//  AI ENGINE
// =====================================================================
struct LetterboxInfo { float scale = 1.0f; int padX = 0; int padY = 0; };

static cv::Mat MakeLetterbox(const cv::Mat& image, int size, LetterboxInfo& info) {
    float sx = static_cast<float>(size) / image.cols;
    float sy = static_cast<float>(size) / image.rows;
    info.scale = std::min(sx, sy);

    int newW = std::max(1, static_cast<int>(std::round(image.cols * info.scale)));
    int newH = std::max(1, static_cast<int>(std::round(image.rows * info.scale)));
    info.padX = (size - newW) / 2;
    info.padY = (size - newH) / 2;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH));
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(info.padX, info.padY, newW, newH)));
    return canvas;
}

struct Detection { cv::Rect box; int classId = -1; float confidence = 0.0f; };

class YoloDetector {
public:
    bool Initialize(const Config& cfg) {
        cfg_ = cfg;
        try {
            net_ = cv::dnn::readNetFromONNX(cfg_.modelPath);
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        } catch (const cv::Exception& e) { return false; }
        return true;
    }

    bool Detect(const cv::Mat& frame, std::vector<Detection>& detections, float& outBestConf) {
        detections.clear();
        outBestConf = 0.0f;
        try {
            LetterboxInfo lb;
            cv::Mat inputImage = MakeLetterbox(frame, cfg_.inputSize, lb);
            cv::Mat blob = cv::dnn::blobFromImage(inputImage, 1.0 / 255.0, cv::Size(cfg_.inputSize, cfg_.inputSize), cv::Scalar(), true, false, CV_32F);
            
            net_.setInput(blob);
            std::vector<cv::Mat> outputs;
            net_.forward(outputs, net_.getUnconnectedOutLayersNames());

            const cv::Mat& tensor = outputs[0];
            const int dim1 = tensor.size[1];
            const int dim2 = tensor.size[2];

            int expectedFeatures = std::min(dim1, dim2);
            int candidates = std::max(dim1, dim2);
            bool featureMajor = (dim1 < dim2); 
            
            int numClasses = expectedFeatures - 4;
            if (numClasses <= 0) return false;

            const float* data = tensor.ptr<float>(0);

            auto valueAt = [&](int feature, int candidate) {
                if (featureMajor) return data[(size_t)feature * candidates + candidate];
                return data[(size_t)candidate * expectedFeatures + feature];
            };

            std::vector<cv::Rect> boxes;
            std::vector<float> scores;
            std::vector<int> classIds;

            for (int c = 0; c < candidates; ++c) {
                float bestScore = -1.0f; int bestClass = -1;
                for (int k = 0; k < numClasses; ++k) {
                    float score = valueAt(4 + k, c);
                    if (std::isfinite(score) && score > bestScore) { bestScore = score; bestClass = k; }
                }

                if (bestClass < 0 || bestScore < cfg_.confidenceThreshold) continue;
                if (bestScore > outBestConf) outBestConf = bestScore;

                float cx = valueAt(0, c), cy = valueAt(1, c), w = valueAt(2, c), h = valueAt(3, c);
                if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(w) || !std::isfinite(h) || w <= 0 || h <= 0) continue;

                float x1 = (cx - w * 0.5f - lb.padX) / lb.scale;
                float y1 = (cy - h * 0.5f - lb.padY) / lb.scale;
                float x2 = (cx + w * 0.5f - lb.padX) / lb.scale;
                float y2 = (cy + h * 0.5f - lb.padY) / lb.scale;

                x1 = std::clamp(x1, 0.0f, (float)frame.cols);
                y1 = std::clamp(y1, 0.0f, (float)frame.rows);
                x2 = std::clamp(x2, 0.0f, (float)frame.cols);
                y2 = std::clamp(y2, 0.0f, (float)frame.rows);

                int left = (int)std::floor(x1), top = (int)std::floor(y1);
                int right = (int)std::ceil(x2), bottom = (int)std::ceil(y2);

                if (right <= left || bottom <= top) continue;

                boxes.emplace_back(left, top, right - left, bottom - top);
                scores.push_back(bestScore);
                classIds.push_back(bestClass);
            }

            for (int classId = 0; classId < numClasses; ++classId) {
                std::vector<cv::Rect> classBoxes; std::vector<float> classScores; std::vector<int> originalIndices;
                for (size_t i = 0; i < boxes.size(); ++i) {
                    if (classIds[i] == classId) {
                        classBoxes.push_back(boxes[i]); classScores.push_back(scores[i]); originalIndices.push_back((int)i);
                    }
                }
                std::vector<int> keep;
                cv::dnn::NMSBoxes(classBoxes, classScores, cfg_.confidenceThreshold, cfg_.nmsThreshold, keep);
                for (int idx : keep) {
                    if (idx >= 0 && idx < (int)originalIndices.size()) {
                        int original = originalIndices[idx];
                        detections.push_back({ boxes[original], classIds[original], scores[original] });
                    }
                }
            }
            return true;
        } catch (...) { return false; }
    }

private:
    Config cfg_;
    cv::dnn::Net net_;
};

// =====================================================================
//  MOUSE & AIM LOGIC
// =====================================================================
void MoveMouse(int targetX, int targetY) {
    INPUT in = {}; in.type = INPUT_MOUSE;
    in.mi.dx = targetX; in.mi.dy = targetY; in.mi.dwFlags = MOUSEEVENTF_MOVE;
    SendInput(1, &in, sizeof(INPUT));
}

// =====================================================================
//  THREADS
// =====================================================================
LRESULT CALLBACK OverlayProc(HWND hwnd, UINT msg, WPARAM w, LPARAM l) {
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    return DefWindowProc(hwnd, msg, w, l);
}

void OverlayThread() {
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = OverlayProc; 
    wc.hInstance = GetModuleHandle(nullptr); 
    wc.lpszClassName = "ModarkOverlay";
    if (!RegisterClass(&wc)) { g_overlayState = -1; return; }
    
    HWND hwnd = CreateWindowEx(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "ModarkOverlay", "", WS_POPUP, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), nullptr, nullptr, wc.hInstance, nullptr);
    
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW); 
    g_overlayHwnd = hwnd; 
    g_overlayState = 1;
    
    MSG msg; 
    while (GetMessage(&msg, nullptr, 0, 0) > 0) { 
        TranslateMessage(&msg); 
        DispatchMessage(&msg); 
    }
}

void CaptureThread(YoloDetector* detector) {
    DesktopCapture capture;
    while (g_appRunning && !capture.Initialize()) std::this_thread::sleep_for(milliseconds(1000));

    cv::Mat raw;
    auto lastTargetTime = Clock::now();
    
    while (g_appRunning) {
        GrabStatus status = capture.Grab(raw);
        
        if (status == GrabStatus::AccessLost) {
            capture.Cleanup(); std::this_thread::sleep_for(milliseconds(500));
            while (g_appRunning && !capture.Initialize()) std::this_thread::sleep_for(milliseconds(1000));
            continue;
        } else if (status == GrabStatus::Timeout) {
            continue;
        } else if (status != GrabStatus::Ok) {
            std::this_thread::sleep_for(milliseconds(2)); continue;
        }

        std::vector<Detection> detections;
        float bestConf = 0.0f;
        
        if (detector->Detect(raw, detections, bestConf)) {
            auto now = Clock::now();
            std::lock_guard<std::mutex> lock(targetMutex);
            
            bool found = false;
            double minDist = 999999;
            Detection bestDet;

            int screenCx = raw.cols / 2;
            int screenCy = raw.rows / 2;

            for (auto& d : detections) {
                double dist = std::hypot((d.box.x + d.box.width/2) - screenCx, (d.box.y + d.box.height/2) - screenCy);
                if (dist > g_circleRadius) continue; 
                if (dist < minDist) { minDist = dist; bestDet = d; found = true; }
            }

            if (found) {
                int tx = bestDet.box.x + bestDet.box.width / 2;
                int ty = bestDet.box.y + bestDet.box.height / 2;
                auto dt = duration_cast<microseconds>(now - lastTargetTime).count(); 
                
                if (globalTarget.found && dt > 0 && dt < 200000) {
                    float vx = (float)(tx - globalTarget.x) / dt;
                    float vy = (float)(ty - globalTarget.y) / dt;
                    globalTarget.vx = globalTarget.vx * 0.5f + vx * 0.5f;
                    globalTarget.vy = globalTarget.vy * 0.5f + vy * 0.5f;
                } else {
                    globalTarget.vx = 0; globalTarget.vy = 0;
                }
                globalTarget.x = tx; globalTarget.y = ty;
                globalTarget.boxX = bestDet.box.x; globalTarget.boxY = bestDet.box.y;
                globalTarget.boxW = bestDet.box.width; globalTarget.boxH = bestDet.box.height;
                globalTarget.detectionTime = now;
            }
            globalTarget.found = found;
            globalTarget.conf = bestConf;
            lastTargetTime = now;
        }
    }
    capture.Cleanup();
}

void AimThread() {
    auto lastLiftTime = Clock::now();
    bool altKeyWasDown = false; 
    
    while (g_appRunning) {
        auto now = Clock::now();
        bool altKeyIsDown = g_isAltPressed; 
        
        if (altKeyIsDown) {
            TargetData t;
            { std::lock_guard<std::mutex> lock(targetMutex); t = globalTarget; }

            if (!altKeyWasDown) {
                if (t.found) {
                    std::cout << "[ALT PRESSED] Target Found! Conf: " << (int)(t.conf * 100) << "% | Processing...\n";
                } else {
                    std::cout << "[ALT PRESSED] No enemy found inside the circle.\n";
                }
            }

            if (t.found) {
                long latency_us = duration_cast<microseconds>(now - t.detectionTime).count();
                float predictX = t.x + (t.vx * latency_us * cfg.predictionFactor);
                float predictY = t.y + (t.vy * latency_us * cfg.predictionFactor);

                predictX += g_headOffsetX;
                predictY += g_headOffsetY;

                int screenCx = GetSystemMetrics(SM_CXSCREEN) / 2;
                int screenCy = GetSystemMetrics(SM_CYSCREEN) / 2;

                int dx = (int)((predictX - screenCx) * g_aimSpeed);
                int dy = (int)((predictY - screenCy) * g_aimSpeed);

                if (std::abs(dx) > 0 || std::abs(dy) > 0) MoveMouse(dx, dy);

                if (!g_isFiring) {
                    INPUT in = {}; in.type = INPUT_MOUSE; in.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
                    SendInput(1, &in, sizeof(INPUT)); g_isFiring = true;
                    lastLiftTime = now - milliseconds(g_liftIntervalMs);
                    std::cout << "[AIM] Shooting!\n"; 
                }
                
                if (duration_cast<milliseconds>(now - lastLiftTime).count() >= g_liftIntervalMs) {
                    if (g_liftValue > 0) {
                        INPUT li = {}; li.type = INPUT_MOUSE; li.mi.dy = -g_liftValue; li.mi.dwFlags = MOUSEEVENTF_MOVE;
                        SendInput(1, &li, sizeof(INPUT));
                    }
                    lastLiftTime = now;
                }
            }
        } else {
            if (g_isFiring) {
                INPUT in = {}; in.type = INPUT_MOUSE; in.mi.dwFlags = MOUSEEVENTF_LEFTUP;
                SendInput(1, &in, sizeof(INPUT)); g_isFiring = false;
                std::cout << "[AIM] Stopped shooting (ALT released).\n"; 
            }
        }
        
        altKeyWasDown = altKeyIsDown; 
        std::this_thread::sleep_for(microseconds(500)); 
    }
}

BOOL WINAPI ConsoleHandler(DWORD sig) {
    if (sig == CTRL_C_EVENT || sig == CTRL_CLOSE_EVENT) {
        g_appRunning = false;
        HWND h = g_overlayHwnd; if (h) PostMessage(h, WM_CLOSE, 0, 0);
        return TRUE;
    }
    return FALSE;
}

// =====================================================================
//  MAIN ENTRY POINT
// =====================================================================
int main() {
    SetProcessDPIAware(); 
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "=== AI Aimbot v111 (Auto-Detect + Fixed Overlay) ===\n";
    
    LoadConfig("config.ini", cfg);

    g_circleRadius = cfg.circleRadius; g_aimSpeed = cfg.aimSpeed;
    g_headOffsetX = cfg.headOffsetX; g_headOffsetY = cfg.headOffsetY;
    g_liftValue = cfg.liftValue; g_liftIntervalMs = cfg.liftIntervalMs;

    YoloDetector detector;
    if (!detector.Initialize(cfg)) {
        std::cout << "[-] Failed to load model. Press Enter to exit.\n";
        std::cin.get(); return 1;
    }

    std::cout << "[3/3] Starting Threads...\n";

    std::thread overlay(OverlayThread);
    std::thread capture(CaptureThread, &detector);
    std::thread aimer(AimThread);

    while (g_appRunning) {
        g_isAltPressed = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
        
        if (GetAsyncKeyState(VK_F8) & 0x8000) { 
            g_appRunning = false; 
            HWND h = g_overlayHwnd; if (h) PostMessage(h, WM_CLOSE, 0, 0); 
            break; 
        }
        
        if (GetAsyncKeyState(VK_UP) & 0x8000) { g_headOffsetY--; std::this_thread::sleep_for(milliseconds(50)); }
        if (GetAsyncKeyState(VK_DOWN) & 0x8000) { g_headOffsetY++; std::this_thread::sleep_for(milliseconds(50)); }
        if (GetAsyncKeyState(VK_LEFT) & 0x8000) { g_headOffsetX--; std::this_thread::sleep_for(milliseconds(50)); }
        if (GetAsyncKeyState(VK_RIGHT) & 0x8000) { g_headOffsetX++; std::this_thread::sleep_for(milliseconds(50)); }
        if (GetAsyncKeyState(VK_ADD) & 0x8000) { g_circleRadius += 5; std::this_thread::sleep_for(milliseconds(50)); }
        if (GetAsyncKeyState(VK_SUBTRACT) & 0x8000) { g_circleRadius -= 5; std::this_thread::sleep_for(milliseconds(50)); }

        HWND h = g_overlayHwnd;
        if (h) {
            HDC hdc = GetDC(h);
            RECT client; GetClientRect(h, &client);
            HBRUSH bg = CreateSolidBrush(RGB(0,0,0)); FillRect(hdc, &client, bg); DeleteObject(bg);

            int cx = GetSystemMetrics(SM_CXSCREEN)/2, cy = GetSystemMetrics(SM_CYSCREEN)/2;
            int r = g_circleRadius;
            
            TargetData t; { std::lock_guard<std::mutex> lock(targetMutex); t = globalTarget; }
            COLORREF color = g_isFiring ? RGB(255,0,0) : t.found ? RGB(255,255,0) : RGB(0,255,0);
            
            HPEN pen = CreatePen(PS_SOLID, 2, color);
            HGDIOBJ oldPen = SelectObject(hdc, pen);
            HGDIOBJ oldBrush = SelectObject(hdc, GetStockObject(NULL_BRUSH));
            Ellipse(hdc, cx - r, cy - r, cx + r, cy + r);
            
            if (t.found) {
                Rectangle(hdc, t.boxX, t.boxY, t.boxX + t.boxW, t.boxY + t.boxH);
            }
            SelectObject(hdc, oldPen); SelectObject(hdc, oldBrush);
            DeleteObject(pen); ReleaseDC(h, hdc);
        }
        std::this_thread::sleep_for(milliseconds(15));
    }

    HWND h = g_overlayHwnd; if (h) PostMessage(h, WM_CLOSE, 0, 0);

    if (overlay.joinable()) overlay.join();
    if (capture.joinable()) capture.join();
    if (aimer.joinable()) aimer.join();
    return 0;
}
