#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <thread>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

// ================= الإعدادات =================
const int FOV = 320;                   // نطاق البحث (سيتم التقاط 640x640 بكسل من المركز)
const int TRIGGER_KEY = VK_MENU;       // زر ALT للتفعيل
const float ASSIST_RADIUS = 300.0f;    // نطاق المساعدة (البكسل)
const float PREDICT_TIME = 0.04f;      // التنبؤ بالحركة
// ============================================

class DXGICapture {
public:
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGIOutputDuplication* deskDupl = nullptr;
    ID3D11Texture2D* acquiredDesktopImage = nullptr;

    bool Init() {
        D3D_FEATURE_LEVEL featureLevel;
        D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &featureLevel, &context);
        IDXGIDevice* dxgiDevice = nullptr; device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice);
        IDXGIAdapter* dxgiAdapter = nullptr; dxgiDevice->GetParent(__uuidof(IDXGIAdapter), (void**)&dxgiAdapter);
        IDXGIOutput* dxgiOutput = nullptr; dxgiAdapter->EnumOutputs(0, &dxgiOutput);
        IDXGIOutput1* dxgiOutput1 = nullptr; dxgiOutput->QueryInterface(__uuidof(IDXGIOutput1), (void**)&dxgiOutput1);
        HRESULT hr = dxgiOutput1->DuplicateOutput(device, &deskDupl);
        dxgiOutput1->Release(); dxgiOutput->Release(); dxgiAdapter->Release(); dxgiDevice->Release();
        return SUCCEEDED(hr);
    }

    bool GetFrame(cv::Mat& outMat) {
        DXGI_OUTDUPL_FRAME_INFO frameInfo; IDXGIResource* desktopResource = nullptr;
        HRESULT hr = deskDupl->AcquireNextFrame(0, &frameInfo, &desktopResource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT || FAILED(hr)) return false;
        desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&acquiredDesktopImage);
        D3D11_TEXTURE2D_DESC desc; acquiredDesktopImage->GetDesc(&desc);
        ID3D11Texture2D* stagingTex = nullptr;
        desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.BindFlags = 0; desc.MiscFlags = 0;
        device->CreateTexture2D(&desc, nullptr, &stagingTex);
        context->CopyResource(stagingTex, acquiredDesktopImage);
        D3D11_MAPPED_SUBRESOURCE mapped; context->Map(stagingTex, 0, D3D11_MAP_READ, 0, &mapped);
        cv::Mat frame(desc.Height, desc.Width, CV_8UC4, mapped.pData, mapped.RowPitch);
        
        int centerX = desc.Width / 2; int centerY = desc.Height / 2;
        cv::Rect roi(centerX - FOV, centerY - FOV, FOV * 2, FOV * 2);
        if (roi.x >= 0 && roi.y >= 0 && roi.x + roi.width <= desc.Width && roi.y + roi.height <= desc.Height) 
            outMat = frame(roi).clone();

        context->Unmap(stagingTex, 0); stagingTex->Release(); acquiredDesktopImage->Release(); deskDupl->ReleaseFrame();
        return true;
    }
};

void HumanizedMove(int targetX, int targetY) {
    POINT currentPos; GetCursorPos(&currentPos);
    float distance = std::hypot(targetX - currentPos.x, targetY - currentPos.y);
    if (distance < 3.0f) return;

    std::random_device rd; std::mt19937 gen(rd());
    std::uniform_int_distribution<> stepsDist(10, 15);
    std::uniform_int_distribution<> jitterDist(-2, 2);
    
    int steps = stepsDist(gen);
    int overshootX = targetX + (rand() % 6 - 3);
    int overshootY = targetY + (rand() % 6 - 3);

    for (int i = 1; i <= steps; ++i) {
        float t = (float)i / steps; 
        float easeT = t * t * (3.0f - 2.0f * t); 
        
        int nextX = currentPos.x + (int)((overshootX - currentPos.x) * easeT);
        int nextY = currentPos.y + (int)((overshootY - currentPos.y) * easeT);

        if (i == steps) { nextX += jitterDist(gen); nextY += jitterDist(gen); }

        INPUT input = { 0 }; input.type = INPUT_MOUSE;
        input.mi.dx = nextX - currentPos.x; input.mi.dy = nextY - currentPos.y; 
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        SendInput(1, &input, sizeof(INPUT));
        std::this_thread::sleep_for(std::chrono::microseconds(300));
    }
}

int main() {
    std::cout << "[+] Loading AI Model (YOLOv8)...\n";
    cv::dnn::Net net = cv::dnn::readNetFromONNX("yolov8n.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU); // معالج لضمان الاستقرار

    DXGICapture capture;
    if (!capture.Init()) { std::cout << "[-] DXGI Failed. Make sure BlueStacks is Windowed.\n"; return 1; }

    cv::Mat frame;
    float lastTargetX = -1, lastTargetY = -1;
    auto lastTime = std::chrono::high_resolution_clock::now();
    int noTargetCounter = 0;

    std::cout << "[+] Aimbot Running. Hold [ALT] to aim.\n\n";

    while (true) {
        if (!capture.GetFrame(frame)) continue;

        cv::Mat rgbFrame; cv::cvtColor(frame, rgbFrame, cv::COLOR_BGRA2RGB);
        cv::Mat blob; cv::dnn::blobFromImage(rgbFrame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);
        std::vector<cv::Mat> outputs; net.forward(outputs, net.getUnconnectedOutLayersNames());

        float* data = (float*)outputs[0].data;
        int dimensions = outputs[0].size[1]; 
        int rows = outputs[0].size[2];       
        float minDist = FLT_MAX; cv::Point bestTarget(-1, -1);
        float bestConf = 0.0f;

        for (int i = 0; i < rows; ++i) {
            float* rowData = data + i * dimensions;
            float confidence = rowData[4]; 
            
            // نسبة الثقة 25% لاكتشاف اللاعبين
            if (confidence > 0.25) { 
                int cx = (int)(rowData[0] * frame.cols);
                int cy = (int)(rowData[1] * frame.rows);
                int w = (int)(rowData[2] * frame.cols);
                int h = (int)(rowData[3] * frame.rows);
                int topY = cy - h / 2; 
                
                float dist = std::hypot(cx - (frame.cols / 2), topY - (frame.rows / 2));
                
                // إذا كان الهدف داخل نطاق المساعدة، اختره
                if (dist < minDist && dist < ASSIST_RADIUS) { 
                    minDist = dist; 
                    bestTarget = cv::Point(cx, topY); 
                    bestConf = confidence;
                }
            }
        }

        // === ميزة التشخيص (الطباعة في الشاشة السوداء) ===
        if (bestTarget.x != -1) {
            std::cout << "[+] Target Found! Confidence: " << bestConf << " | Distance: " << minDist << std::endl;
            noTargetCounter = 0;
        } else {
            noTargetCounter++;
            if (noTargetCounter > 100) { // لتجنب إغراق الشاشة
                std::cout << "[-] Scanning... No targets in range.\n";
                noTargetCounter = 0;
            }
        }

        // التحرك والإطلاق (فقط عند الضغط على ALT)
        if ((GetAsyncKeyState(TRIGGER_KEY) & 0x8000) && bestTarget.x != -1) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            
            float aimX = bestTarget.x;
            float aimY = bestTarget.y;

            if (lastTargetX != -1 && deltaTime > 0.01f && deltaTime < 0.1f) {
                float velX = (bestTarget.x - lastTargetX) / deltaTime;
                float velY = (bestTarget.y - lastTargetY) / deltaTime;
                aimX += velX * PREDICT_TIME;
                aimY += velY * PREDICT_TIME;
            }

            lastTargetX = bestTarget.x; lastTargetY = bestTarget.y; lastTime = currentTime;

            int actualX = aimX + (GetSystemMetrics(SM_CXSCREEN) / 2) - FOV;
            int actualY = aimY + (GetSystemMetrics(SM_CYSCREEN) / 2) - FOV;

            HumanizedMove(actualX, actualY);

            if (minDist < 25.0f) { // إذا كان التصويب قريباً جداً من الهدف
                int reactionDelay = 20 + (rand() % 30);
                std::this_thread::sleep_for(std::chrono::milliseconds(reactionDelay));
                
                INPUT click = { 0 }; click.type = INPUT_MOUSE; click.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
                SendInput(1, &click, sizeof(INPUT));
                std::this_thread::sleep_for(std::chrono::milliseconds(30)); 
                click.mi.dwFlags = MOUSEEVENTF_LEFTUP; SendInput(1, &click, sizeof(INPUT));
                std::this_thread::sleep_for(std::chrono::milliseconds(60)); 
            }
        } else {
            lastTargetX = -1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return 0;
}
