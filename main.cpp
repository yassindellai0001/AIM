// language: C++, file: main.cpp, runtime: MSVC v143 (C++17), Windows 10/11
// v99 - Fixed 640x640 AI Input, CPU Only, Multi-Threaded

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/ocl.hpp>
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
//  GLOBALS & CONFIG
// =====================================================================
struct Config {
    int circleRadius = 150;
    float aimSpeed = 0.85f;
    int headOffsetY = 0, headOffsetX = 0;
    int liftValue = 3, liftIntervalMs = 30;
    float confThreshold = 0.40f, nmsThreshold = 0.45f;
    float predictionFactor = 0.05f;
    int aiInputSize = 640; // [MODIFIED] Fixed to 640x640
    int windMouse = 1;
    int useGPU = 0; // [MODIFIED] Forced to 0 (CPU)
};
static Config cfg;

struct TargetData {
    bool found = false;
    int x = 0, y = 0;
    float vx = 0.0f, vy = 0.0f;
    float conf = 0.0f;
    int boxX = 0, boxY = 0, boxW = 0, boxH = 0;
};
static std::mutex targetMutex;
static TargetData globalTarget;

static std::atomic<bool> g_appRunning(true);
static std::atomic<bool> g_isAltPressed(false);
static std::atomic<bool> g_isFiring(false);
static std::atomic<int>  g_circleRadius(150);
static std::atomic<float> g_aimSpeed(0.85f);
static std::atomic<int>   g_headOffsetX(0);
static std::atomic<int>   g_headOffsetY(0);
static std::atomic<int>   g_liftValue(3);
static std::atomic<int>   g_liftIntervalMs(30);

static std::atomic<HWND> g_overlayHwnd(nullptr);
static std::atomic<int>  g_overlayState(0);

// =====================================================================
//  AI ENGINE
// =====================================================================
class AIEngine {
    cv::dnn::Net net;
    int inputWidth = 640, inputHeight = 640;
public:
    bool LoadModel(const std::string& modelPath, int size) {
        std::cout << "[1/3] Loading AI Model (" << modelPath << ")...\n";
        try {
            net = cv::dnn::readNetFromONNX(modelPath);
            
            // [FIX] Always use CPU to prevent the Reshape crash
            cv::ocl::setUseOpenCL(false);
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "      -> Mode: CPU (Stable)\n";
            
            inputWidth = size; inputHeight = size;
            std::cout << "      -> Input Size: " << inputWidth << "x" << inputHeight << "\n";
            std::cout << "[2/3] AI Model Loaded Successfully.\n";
            return true;
        } catch (const std::exception& e) { 
            std::cout << "[-] FATAL ERROR: Failed to load AI model!\n";
            std::cout << "[-] Reason: " << e.what() << "\n";
            return false; 
        }
    }

    std::vector<cv::Rect> Detect(cv::Mat& frame, float confThresh, float nmsThresh, float& outBestConf) {
        std::vector<cv::Rect> boxes;
        if (net.empty()) return boxes;

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);
        net.setInput(blob);
        cv::Mat outputs = net.forward();

        cv::Mat outputMat(outputs.size[1], outputs.size[2], CV_32F, outputs.ptr<float>());
        cv::transpose(outputMat, outputMat);

        std::vector<float> confidences;
        float xFactor = (float)frame.cols / inputWidth;
        float yFactor = (float)frame.rows / inputHeight;
        outBestConf = 0.0f;

        for (int i = 0; i < outputMat.rows; ++i) {
            float* row = outputMat.ptr<float>(i);
            float maxConf = 0.0f;
            for (int j = 4; j < outputMat.cols; j++) if (row[j] > maxConf) maxConf = row[j];
            
            if (maxConf < confThresh) continue;
            if (maxConf > outBestConf) outBestConf = maxConf;

            int left = int((row[0] - 0.5 * row[2]) * xFactor);
            int top = int((row[1] - 0.5 * row[3]) * yFactor);
            int width = int(row[2] * xFactor);
            int height = int(row[3] * yFactor);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(maxConf);
        }
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThresh, nmsThresh, indices);
        
        std::vector<cv::Rect> finalBoxes;
        for (int idx : indices) finalBoxes.push_back(boxes[idx]);
        return finalBoxes;
    }
};

// =====================================================================
//  CAPTURE ENGINE
// =====================================================================
enum class CaptureStatus { Success, Timeout, AccessLost, DeviceLost, Error, InvalidROI };

class DXGICaptureEngine {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGIOutputDuplication> deskDupl;
    ComPtr<ID3D11Texture2D> stagingTex;
    int texW = 0, texH = 0;
public:
    bool Init() {
        Cleanup();
        D3D_FEATURE_LEVEL fl;
        if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, device.GetAddressOf(), &fl, context.GetAddressOf()))) return false;
        ComPtr<IDXGIDevice> dxgiDevice; device.As(&dxgiDevice);
        ComPtr<IDXGIAdapter> dxgiAdapter; dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)dxgiAdapter.GetAddressOf());
        ComPtr<IDXGIOutput> dxgiOutput; dxgiAdapter->EnumOutputs(0, dxgiOutput.GetAddressOf());
        ComPtr<IDXGIOutput1> dxgiOutput1; dxgiOutput.As(&dxgiOutput1);
        return SUCCEEDED(dxgiOutput1->DuplicateOutput(device.Get(), deskDupl.GetAddressOf()));
    }

    CaptureStatus GrabFrame(cv::Mat& outFrame, int currentRadius) {
        if (!deskDupl) return CaptureStatus::Error;
        DXGI_OUTDUPL_FRAME_INFO info = {};
        ComPtr<IDXGIResource> resource;
        HRESULT hr = deskDupl->AcquireNextFrame(1, &info, resource.GetAddressOf());
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) return CaptureStatus::Timeout;
        if (hr == DXGI_ERROR_ACCESS_LOST) return CaptureStatus::AccessLost;
        if (FAILED(hr)) return CaptureStatus::Error;

        ComPtr<ID3D11Texture2D> acquired; resource.As(&acquired);
        D3D11_TEXTURE2D_DESC desc; acquired->GetDesc(&desc);

        int cx = desc.Width / 2, cy = desc.Height / 2;
        int fov = currentRadius + 20;
        int cropW = fov * 2, cropH = fov * 2;
        int cropX = std::max(0, cx - fov), cropY = std::max(0, cy - fov);
        if (cropX + cropW > (int)desc.Width) cropW = desc.Width - cropX;
        if (cropY + cropH > (int)desc.Height) cropH = desc.Height - cropY;

        if (!stagingTex || texW != cropW || texH != cropH) {
            stagingTex.Reset();
            D3D11_TEXTURE2D_DESC sdesc = {};
            sdesc.Width = cropW; sdesc.Height = cropH; sdesc.MipLevels = 1; sdesc.ArraySize = 1;
            sdesc.Format = desc.Format; sdesc.SampleDesc.Count = 1;
            sdesc.Usage = D3D11_USAGE_STAGING; sdesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
            device->CreateTexture2D(&sdesc, nullptr, stagingTex.GetAddressOf());
            texW = cropW; texH = cropH;
        }

        D3D11_BOX box = { (UINT)cropX, (UINT)cropY, 0, (UINT)(cropX + cropW), (UINT)(cropY + cropH), 1 };
        context->CopySubresourceRegion(stagingTex.Get(), 0, 0, 0, 0, acquired.Get(), 0, &box);
        deskDupl->ReleaseFrame();

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        if (FAILED(context->Map(stagingTex.Get(), 0, D3D11_MAP_READ, 0, &mapped))) return CaptureStatus::Error;

        cv::Mat full(cropH, cropW, CV_8UC4, mapped.pData, mapped.RowPitch);
        full.copyTo(outFrame);
        context->Unmap(stagingTex.Get(), 0);
        return CaptureStatus::Success;
    }
    void Cleanup() { stagingTex.Reset(); deskDupl.Reset(); context.Reset(); device.Reset(); }
};

// =====================================================================
//  WINDDMOUSE ALGORITHM
// =====================================================================
void WindMouse(int targetX, int targetY) {
    if (cfg.windMouse == 0) {
        INPUT in = {}; in.type = INPUT_MOUSE;
        in.mi.dx = targetX; in.mi.dy = targetY;
        in.mi.dwFlags = MOUSEEVENTF_MOVE;
        SendInput(1, &in, sizeof(INPUT));
        return;
    }

    double dist = std::hypot(targetX, targetY);
    if (dist < 1.0) return;

    double M = 15.0, W = 5.0, D = 12.0;
    double sqrt3 = std::sqrt(3.0), sqrt5 = std::sqrt(5.0);
    
    static double windX = 0.0, windY = 0.0;
    double vX = 0.0, vY = 0.0;
    double curX = 0.0, curY = 0.0;
    double tDist = dist;

    while (tDist >= 1.0) {
        double w = std::min(W / dist, 1.0);
        windX = windX / sqrt3 + (std::rand() % 100 / 100.0 * 2.0 - 1.0) * w;
        windY = windY / sqrt3 + (std::rand() % 100 / 100.0 * 2.0 - 1.0) * w;
        
        windX = windX / sqrt5; windY = windY / sqrt5;

        double vX_new = vX + windX + (targetX - curX) / dist * M;
        double vY_new = vY + windY + (targetY - curY) / dist * M;
        
        double vMag = std::hypot(vX_new, vY_new);
        double maxV = std::min(D, tDist);
        
        if (vMag > maxV) {
            vX = vX_new / vMag * maxV;
            vY = vY_new / vMag * maxV;
        } else {
            vX = vX_new; vY = vY_new;
        }

        double stepX = vX, stepY = vY;
        if (std::abs(stepX) > 0.0 || std::abs(stepY) > 0.0) {
            curX += stepX; curY += stepY;
            INPUT in = {}; in.type = INPUT_MOUSE;
            in.mi.dx = (int)stepX; in.mi.dy = (int)stepY;
            in.mi.dwFlags = MOUSEEVENTF_MOVE;
            SendInput(1, &in, sizeof(INPUT));
            
            tDist = std::hypot(targetX - curX, targetY - curY);
            std::this_thread::sleep_for(std::chrono::microseconds(100 + (rand() % 300)));
        } else break;
    }
}

// =====================================================================
//  THREADS
// =====================================================================
void OverlayThread() {
    WNDCLASS wc = { 0 };
    wc.lpfnWndProc = DefWindowProc; wc.hInstance = GetModuleHandle(nullptr); wc.lpszClassName = "ModarkOverlay";
    if (!RegisterClass(&wc)) { g_overlayState = -1; return; }
    
    HWND hwnd = CreateWindowEx(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
        "ModarkOverlay", "", WS_POPUP, 0, 0, GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN), nullptr, nullptr, wc.hInstance, nullptr);
    
    SetLayeredWindowAttributes(hwnd, RGB(0, 0, 0), 0, LWA_COLORKEY);
    ShowWindow(hwnd, SW_SHOW);
    g_overlayHwnd = hwnd;
    g_overlayState = 1;

    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessage(&msg); }
}

void CaptureThread(AIEngine* ai) {
    DXGICaptureEngine capture;
    while (g_appRunning && !capture.Init()) std::this_thread::sleep_for(milliseconds(1000));

    cv::Mat raw, bgr, circleMask;
    int lastMaskRadius = -1;
    auto lastTargetTime = Clock::now();
    
    while (g_appRunning) {
        int radius = g_circleRadius;
        int cx = radius + 20, cy = radius + 20;

        if (capture.GrabFrame(raw, radius) == CaptureStatus::Success) {
            cv::cvtColor(raw, bgr, cv::COLOR_RGBA2BGR);
            
            if (radius != lastMaskRadius) {
                circleMask = cv::Mat::zeros(bgr.size(), CV_8UC1);
                cv::circle(circleMask, cv::Point(cx, cy), radius, cv::Scalar(255), -1);
                lastMaskRadius = radius;
            }
            
            cv::Mat masked; bgr.copyTo(masked, circleMask);
            float bestConf = 0.0f;
            auto boxes = ai->Detect(masked, cfg.confThreshold, cfg.nmsThreshold, bestConf);

            std::lock_guard<std::mutex> lock(targetMutex);
            auto now = Clock::now();
            
            bool found = false;
            double minDist = 999999;
            cv::Rect bestBox;

            for (auto& box : boxes) {
                double d = std::hypot((box.x + box.width/2) - cx, (box.y + box.height/2) - cy);
                if (d > radius) continue;
                if (d < minDist) { minDist = d; bestBox = box; found = true; }
            }

            if (found) {
                int tx = bestBox.x + bestBox.width / 2;
                int ty = bestBox.y + bestBox.height / 2;
                auto dt = duration_cast<milliseconds>(now - lastTargetTime).count();
                
                if (globalTarget.found && dt > 0 && dt < 200) {
                    float vx = (float)(tx - globalTarget.x) / dt;
                    float vy = (float)(ty - globalTarget.y) / dt;
                    globalTarget.vx = globalTarget.vx * 0.4f + vx * 0.6f;
                    globalTarget.vy = globalTarget.vy * 0.4f + vy * 0.6f;
                } else {
                    globalTarget.vx = 0; globalTarget.vy = 0;
                }
                globalTarget.x = tx; globalTarget.y = ty;
                globalTarget.boxX = bestBox.x; globalTarget.boxY = bestBox.y;
                globalTarget.boxW = bestBox.width; globalTarget.boxH = bestBox.height;
            }
            globalTarget.found = found;
            globalTarget.conf = bestConf;
            lastTargetTime = now;
        } else {
            std::this_thread::sleep_for(milliseconds(1));
        }
    }
    capture.Cleanup();
}

void AimThread() {
    auto lastLiftTime = Clock::now();
    auto lastLockTime = Clock::now();
    bool wasLocked = false;
    
    while (g_appRunning) {
        auto now = Clock::now();
        if (g_isAltPressed) {
            TargetData t;
            { std::lock_guard<std::mutex> lock(targetMutex); t = globalTarget; }

            if (t.found) {
                if (!wasLocked) { lastLockTime = now; wasLocked = true; }
                
                if (duration_cast<milliseconds>(now - lastLockTime).count() >= 30) {
                    float px = t.x + t.vx * 50.0f * cfg.predictionFactor + g_headOffsetX;
                    float py = t.y + t.vy * 50.0f * cfg.predictionFactor + g_headOffsetY;

                    int dx = (int)((px - (g_circleRadius + 20)) * g_aimSpeed);
                    int dy = (int)((py - (g_circleRadius + 20)) * g_aimSpeed);

                    if (std::abs(dx) > 0 || std::abs(dy) > 0) {
                        WindMouse(dx, dy);
                    }

                    if (!g_isFiring) {
                        INPUT in = {}; in.type = INPUT_MOUSE; in.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
                        SendInput(1, &in, sizeof(INPUT)); g_isFiring = true;
                        lastLiftTime = now - milliseconds(g_liftIntervalMs);
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
                wasLocked = false;
            }
        } else {
            if (g_isFiring) {
                INPUT in = {}; in.type = INPUT_MOUSE; in.mi.dwFlags = MOUSEEVENTF_LEFTUP;
                SendInput(1, &in, sizeof(INPUT)); g_isFiring = false;
            }
            wasLocked = false;
        }
        std::this_thread::sleep_for(milliseconds(1));
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
//  MAIN
// =====================================================================
int main() {
    SetProcessDPIAware(); SetConsoleCtrlHandler(ConsoleHandler, TRUE);

    std::cout << "[0/3] Starting Aimbot (640x640)...\n";

    // Hardcoded configs to ensure it works without INI file
    // (You can still uncomment the reading part if you want)
    g_circleRadius = cfg.circleRadius; g_aimSpeed = cfg.aimSpeed;
    g_headOffsetX = cfg.headOffsetX; g_headOffsetY = cfg.headOffsetY;
    g_liftValue = cfg.liftValue; g_liftIntervalMs = cfg.liftIntervalMs;

    AIEngine ai;
    if (!ai.LoadModel("best.onnx", cfg.aiInputSize)) {
        std::cout << "\n[-] PROGRAM CRASHED DURING AI LOADING. Press Enter to exit...\n";
        std::cin.get();
        return 1;
    }

    std::cout << "[3/3] Starting Threads...\n";

    std::thread overlay(OverlayThread);
    std::thread capture(CaptureThread, &ai);
    std::thread aimer(AimThread);

    while (g_appRunning) {
        g_isAltPressed = (GetAsyncKeyState(VK_MENU) & 0x8000) != 0;
        
        if (GetAsyncKeyState(VK_F8) & 0x8000) { g_appRunning = false; break; }
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
                int fov = r + 20;
                int drawX = (cx - fov) + t.boxX;
                int drawY = (cy - fov) + t.boxY;
                Rectangle(hdc, drawX, drawY, drawX + t.boxW, drawY + t.boxH);
            }
            
            SelectObject(hdc, oldPen); SelectObject(hdc, oldBrush);
            DeleteObject(pen); ReleaseDC(h, hdc);
        }
        std::this_thread::sleep_for(milliseconds(15));
    }

    if (overlay.joinable()) overlay.join();
    if (capture.joinable()) capture.join();
    if (aimer.joinable()) aimer.join();
    
    std::cout << "[+] Program Ended Successfully.\n";
    std::cout << "Press Enter to exit...\n";
    std::cin.get();
    return 0;
}
