/**
 * HKB-BV: Hybrid Knowledge-Based Biometric Verification
 * ESP32 Main Sketch
 * 
 * Hardware:
 *   - ESP32-WROOM-32 (240MHz, 520KB SRAM)
 *   - R307 Optical Fingerprint Sensor (500 dpi)
 * 
 * Dependencies (install via Arduino Library Manager):
 *   - Adafruit Fingerprint Sensor Library (v2.1.0+)
 *   - PubSubClient (v2.8.0+)  MQTT client
 *   - ArduinoJson (v6.21.0+)  JSON serialisation
 *   - WiFiClientSecure        TLS support (built-in ESP32)
 *   - mbedTLS                 Cryptography (built-in ESP32)
 */

// ============================================================
// Includes
// ============================================================
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <HardwareSerial.h>
#include <Adafruit_Fingerprint.h>

#include "minutiae_extractor.hpp"
#include "fuzzy_liveness.hpp"
#include "communication.hpp"

// ============================================================
// Configuration — Edit before flashing
// ============================================================

// WiFi credentials
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// MQTT broker
const char* MQTT_HOST     = "mqtt.hkb-bv.local";
const int   MQTT_PORT     = 8883;     // TLS port
const char* MQTT_USER     = "esp32_client";
const char* MQTT_PASS     = "YOUR_MQTT_PASSWORD";
const char* MQTT_CLIENT   = "esp32_hkb_bv_001";

// MQTT topics
const char* TOPIC_PROBE   = "hkb_bv/probe";
const char* TOPIC_RESULT  = "hkb_bv/result";
const char* TOPIC_STATUS  = "hkb_bv/status";
const char* TOPIC_ALERT   = "hkb_bv/alert";

// Hardware pins
#define FINGERPRINT_RX_PIN  16
#define FINGERPRINT_TX_PIN  17
#define LED_GREEN_PIN       25
#define LED_RED_PIN         26
#define BUZZER_PIN          27
#define STATUS_LED_PIN       2

// Timing (milliseconds)
#define VERIFY_TIMEOUT_MS   10000
#define MQTT_TIMEOUT_MS     5000
#define WIFI_TIMEOUT_MS     15000
#define LIVENESS_TIMEOUT_MS 3000

// Firmware version
#define FIRMWARE_VERSION    "1.0.0"
#define DEVICE_ID           "HKB_BV_ESP32_001"

// ============================================================
// TLS Certificates (replace with actual certificates)
// ============================================================

// CA Certificate (Root CA that signed the MQTT broker cert)
const char* CA_CERT = R"EOF(
-----BEGIN CERTIFICATE-----
YOUR_CA_CERTIFICATE_HERE
-----END CERTIFICATE-----
)EOF";

// Client Certificate (for mTLS)
const char* CLIENT_CERT = R"EOF(
-----BEGIN CERTIFICATE-----
YOUR_CLIENT_CERTIFICATE_HERE
-----END CERTIFICATE-----
)EOF";

// Client Private Key (for mTLS)
const char* CLIENT_KEY = R"EOF(
-----BEGIN RSA PRIVATE KEY-----
YOUR_CLIENT_PRIVATE_KEY_HERE
-----END RSA PRIVATE KEY-----
)EOF";

// ============================================================
// Global Objects
// ============================================================

// Fingerprint sensor on UART2
HardwareSerial fingerprintSerial(2);
Adafruit_Fingerprint fingerprintSensor = Adafruit_Fingerprint(&fingerprintSerial);

// TLS WiFi client
WiFiClientSecure wifiClient;

// MQTT client
PubSubClient mqttClient(wifiClient);

// HKB-BV modules
MinutiaeExtractor minutiaeExtractor;
FuzzyLiveness     fuzzyLiveness;
Communication     comm;

// State machine
enum class SystemState {
    IDLE,
    SCANNING,
    EXTRACTING,
    TRANSMITTING,
    WAITING_RESULT,
    ACCEPTED,
    REJECTED,
    ERROR
};

SystemState currentState = SystemState::IDLE;

// Verification result
struct VerificationResult {
    bool        decided;
    int         decision;         // 0 = reject, 1 = accept
    float       belief;
    float       similarity_score;
    float       liveness_score;
    String      reason;
    String      audit_log_id;
    unsigned long latency_ms;
};

VerificationResult lastResult;

// Session tracking
unsigned long sessionStartMs   = 0;
unsigned long lastHeartbeatMs  = 0;
static int    verificationCount = 0;

// ============================================================
// Setup
// ============================================================

void setup() {
    Serial.begin(115200);
    delay(100);
    
    Serial.println("========================================");
    Serial.println(" HKB-BV Edge Device v" FIRMWARE_VERSION);
    Serial.println(" Device ID: " DEVICE_ID);
    Serial.println("========================================");
    
    // Initialise GPIO
    initGPIO();
    
    // Initialise fingerprint sensor
    if (!initFingerprintSensor()) {
        Serial.println("[ERROR] Fingerprint sensor init failed");
        indicateError();
        while (true) { delay(1000); }
    }
    
    // Initialise modules
    minutiaeExtractor.begin();
    fuzzyLiveness.begin();
    
    // Connect to WiFi
    if (!connectWiFi()) {
        Serial.println("[ERROR] WiFi connection failed");
        indicateError();
        while (true) { delay(1000); }
    }
    
    // Configure TLS
    configTLS();
    
    // Connect to MQTT broker
    mqttClient.setServer(MQTT_HOST, MQTT_PORT);
    mqttClient.setCallback(mqttCallback);
    mqttClient.setKeepAlive(60);
    mqttClient.setSocketTimeout(10);
    
    if (!connectMQTT()) {
        Serial.println("[ERROR] MQTT connection failed");
        indicateError();
    }
    
    // Publish startup status
    publishStatus("online", "HKB-BV edge device ready");
    
    // Indicate ready
    indicateReady();
    
    Serial.println("[INFO] System initialised. Awaiting fingerprint...");
}

// ============================================================
// Main Loop
// ============================================================

void loop() {
    // Maintain MQTT connection
    if (!mqttClient.connected()) {
        Serial.println("[WARN] MQTT disconnected. Reconnecting...");
        connectMQTT();
    }
    mqttClient.loop();
    
    // Heartbeat every 30 seconds
    if (millis() - lastHeartbeatMs > 30000) {
        publishHeartbeat();
        lastHeartbeatMs = millis();
    }
    
    // State machine
    switch (currentState) {
        case SystemState::IDLE:
            handleIdle();
            break;
        
        case SystemState::SCANNING:
            handleScanning();
            break;
        
        case SystemState::EXTRACTING:
            handleExtracting();
            break;
        
        case SystemState::TRANSMITTING:
            handleTransmitting();
            break;
        
        case SystemState::WAITING_RESULT:
            handleWaitingResult();
            break;
        
        case SystemState::ACCEPTED:
            handleAccepted();
            break;
        
        case SystemState::REJECTED:
            handleRejected();
            break;
        
        case SystemState::ERROR:
            handleError();
            break;
    }
}

// ============================================================
// State Handlers
// ============================================================

void handleIdle() {
    // Wait for finger placement
    uint8_t sensorState = fingerprintSensor.getImage();
    
    if (sensorState == FINGERPRINT_OK) {
        Serial.println("[INFO] Finger detected. Starting scan...");
        sessionStartMs = millis();
        currentState = SystemState::SCANNING;
        indicateLED(LED_GREEN_PIN, true);
    }
}

void handleScanning() {
    // Convert image to feature buffer
    uint8_t result = fingerprintSensor.image2Tz(1);
    
    if (result == FINGERPRINT_OK) {
        Serial.println("[INFO] Image converted. Extracting features...");
        currentState = SystemState::EXTRACTING;
    } else {
        Serial.println("[ERROR] Image conversion failed");
        currentState = SystemState::ERROR;
    }
}

void handleExtracting() {
    // Step 1: Extract minutiae features
    MinutiaeFeatures features = minutiaeExtractor.extract(fingerprintSensor);
    
    if (!features.valid) {
        Serial.println("[ERROR] Minutiae extraction failed");
        currentState = SystemState::ERROR;
        return;
    }
    
    Serial.println("[INFO] Minutiae extracted successfully");
    
    // Step 2: Fuzzy liveness detection
    LivenessResult liveness = fuzzyLiveness.assess(features);
    
    Serial.printf("[INFO] Liveness score: %.3f (threshold: %.3f)\n",
                  liveness.score, LIVENESS_THRESHOLD);
    
    if (liveness.score < LIVENESS_THRESHOLD) {
        Serial.println("[WARN] Liveness check FAILED — spoof detected");
        
        // Log spoof attempt
        publishAlert("spoof_detected", liveness.score);
        
        // Reject immediately
        lastResult = {
            .decided      = true,
            .decision     = 0,
            .belief       = 0.0f,
            .liveness_score = liveness.score,
            .reason       = "Liveness check failed",
            .latency_ms   = millis() - sessionStartMs
        };
        
        currentState = SystemState::REJECTED;
        return;
    }
    
    // Step 3: Prepare probe payload for backend
    // Store features in static buffer for transmission
    comm.setProbeFeatures(features, liveness.score);
    
    currentState = SystemState::TRANSMITTING;
}

void handleTransmitting() {
    // Build JSON payload
    StaticJsonDocument<2048> doc;
    
    doc["device_id"]       = DEVICE_ID;
    doc["candidate_id"]    = comm.getCandidateId();
    doc["liveness_score"]  = comm.getLivenessScore();
    doc["session_id"]      = comm.getSessionId();
    doc["timestamp"]       = millis();
    doc["firmware_version"]= FIRMWARE_VERSION;
    
    // Embed encrypted embedding (base64-encoded)
    doc["fingerprint_embedding"] = comm.getEncryptedEmbedding();
    
    // Serialise
    String payload;
    serializeJson(doc, payload);
    
    // Publish to MQTT broker
    bool published = mqttClient.publish(TOPIC_PROBE, payload.c_str(), true);
    
    if (published) {
        Serial.println("[INFO] Probe published to backend");
        currentState = SystemState::WAITING_RESULT;
    } else {
        Serial.println("[ERROR] Failed to publish probe");
        currentState = SystemState::ERROR;
    }
}

void handleWaitingResult() {
    // Wait for backend response via MQTT callback
    unsigned long waitStart = millis();
    
    while (!lastResult.decided && (millis() - waitStart < MQTT_TIMEOUT_MS)) {
        mqttClient.loop();
        delay(10);
    }
    
    if (!lastResult.decided) {
        Serial.println("[ERROR] Backend response timeout");
        currentState = SystemState::ERROR;
    }
}

void handleAccepted() {
    unsigned long latency = millis() - sessionStartMs;
    
    Serial.printf("[INFO] VERIFICATION ACCEPTED (belief=%.3f, latency=%lums)\n",
                  lastResult.belief, latency);
    
    indicateAccepted();
    
    verificationCount++;
    
    // Publish confirmation
    publishStatus("accepted", "Candidate verified successfully");
    
    // Reset state
    delay(2000);
    resetSession();
    currentState = SystemState::IDLE;
}

void handleRejected() {
    unsigned long latency = millis() - sessionStartMs;
    
    Serial.printf("[WARN] VERIFICATION REJECTED (reason=%s, latency=%lums)\n",
                  lastResult.reason.c_str(), latency);
    
    indicateRejected();
    
    // Publish rejection with reason
    publishStatus("rejected", lastResult.reason);
    
    // Reset state
    delay(2000);
    resetSession();
    currentState = SystemState::IDLE;
}

void handleError() {
    Serial.println("[ERROR] Error state — resetting session");
    
    indicateError();
    publishStatus("error", "System error occurred");
    
    delay(2000);
    resetSession();
    currentState = SystemState::IDLE;
}

// ============================================================
// MQTT Callback
// ============================================================

void mqttCallback(char* topic, byte* payload, unsigned int length) {
    String topicStr = String(topic);
    String payloadStr = "";
    
    for (unsigned int i = 0; i < length; i++) {
        payloadStr += (char)payload[i];
    }
    
    Serial.printf("[MQTT] Message received on %s\n", topic);
    
    if (topicStr == TOPIC_RESULT) {
        handleVerificationResult(payloadStr);
    }
}

void handleVerificationResult(const String& payload) {
    StaticJsonDocument<1024> doc;
    DeserializationError err = deserializeJson(doc, payload);
    
    if (err) {
        Serial.println("[ERROR] Failed to parse result JSON");
        currentState = SystemState::ERROR;
        return;
    }
    
    int   decision        = doc["decision"]        | 0;
    float belief          = doc["belief"]           | 0.0f;
    float similarity      = doc["similarity_score"] | 0.0f;
    const char* reason    = doc["reason"]           | "Unknown";
    const char* auditId   = doc["audit_log_id"]     | "";
    
    lastResult = {
        .decided        = true,
        .decision       = decision,
        .belief         = belief,
        .similarity_score = similarity,
        .liveness_score = comm.getLivenessScore(),
        .reason         = String(reason),
        .audit_log_id   = String(auditId),
        .latency_ms     = millis() - sessionStartMs
    };
    
    currentState = (decision == 1) ? SystemState::ACCEPTED : SystemState::REJECTED;
}

// ============================================================
// WiFi & MQTT Helpers
// ============================================================

bool connectWiFi() {
    Serial.printf("[INFO] Connecting to WiFi: %s\n", WIFI_SSID);
    
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    
    unsigned long startMs = millis();
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - startMs > WIFI_TIMEOUT_MS) {
            Serial.println("[ERROR] WiFi timeout");
            return false;
        }
        delay(500);
        Serial.print(".");
    }
    
    Serial.printf("\n[INFO] WiFi connected. IP: %s\n",
                  WiFi.localIP().toString().c_str());
    return true;
}

void configTLS() {
    wifiClient.setCACert(CA_CERT);
    wifiClient.setCertificate(CLIENT_CERT);  // mTLS
    wifiClient.setPrivateKey(CLIENT_KEY);    // mTLS
    
    Serial.println("[INFO] TLS 1.3 configured (mTLS enabled)");
}

bool connectMQTT() {
    Serial.printf("[INFO] Connecting to MQTT: %s:%d\n", MQTT_HOST, MQTT_PORT);
    
    int retries = 0;
    while (!mqttClient.connected() && retries < 3) {
        if (mqttClient.connect(MQTT_CLIENT, MQTT_USER, MQTT_PASS)) {
            Serial.println("[INFO] MQTT connected");
            mqttClient.subscribe(TOPIC_RESULT);
            return true;
        }
        Serial.printf("[WARN] MQTT connect failed (state=%d). Retry %d/3\n",
                      mqttClient.state(), retries + 1);
        retries++;
        delay(2000);
    }
    
    return false;
}

// ============================================================
// GPIO & Indicator Helpers
// ============================================================

void initGPIO() {
    pinMode(LED_GREEN_PIN,  OUTPUT);
    pinMode(LED_RED_PIN,    OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    pinMode(BUZZER_PIN,     OUTPUT);
    
    digitalWrite(LED_GREEN_PIN,  LOW);
    digitalWrite(LED_RED_PIN,    LOW);
    digitalWrite(STATUS_LED_PIN, LOW);
    digitalWrite(BUZZER_PIN,     LOW);
    
    Serial.println("[INFO] GPIO initialised");
}

void indicateLED(int pin, bool state) {
    digitalWrite(pin, state ? HIGH : LOW);
}

void indicateReady() {
    digitalWrite(STATUS_LED_PIN, HIGH);
    tone(BUZZER_PIN, 1000, 200);
    delay(200);
    noTone(BUZZER_PIN);
}

void indicateAccepted() {
    digitalWrite(LED_GREEN_PIN, HIGH);
    tone(BUZZER_PIN, 1500, 200);
    delay(300);
    noTone(BUZZER_PIN);
    delay(300);
}

void indicateRejected() {
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_RED_PIN, HIGH);
        tone(BUZZER_PIN, 500, 200);
        delay(200);
        digitalWrite(LED_RED_PIN, LOW);
        noTone(BUZZER_PIN);
        delay(200);
    }
}

void indicateError() {
    for (int i = 0; i < 5; i++) {
        digitalWrite(LED_RED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_RED_PIN, LOW);
        delay(100);
    }
}

// ============================================================
// MQTT Publishing Helpers
// ============================================================

void publishStatus(const char* status, const char* message) {
    StaticJsonDocument<256> doc;
    doc["device_id"] = DEVICE_ID;
    doc["status"]    = status;
    doc["message"]   = message;
    doc["uptime_ms"] = millis();
    doc["count"]     = verificationCount;
    
    String payload;
    serializeJson(doc, payload);
    
    mqttClient.publish(TOPIC_STATUS, payload.c_str(), false);
}

void publishAlert(const char* alert_type, float score) {
    StaticJsonDocument<256> doc;
    doc["device_id"]  = DEVICE_ID;
    doc["alert_type"] = alert_type;
    doc["score"]      = score;
    doc["timestamp"]  = millis();
    
    String payload;
    serializeJson(doc, payload);
    
    mqttClient.publish(TOPIC_ALERT, payload.c_str(), false);
}

void publishHeartbeat() {
    StaticJsonDocument<256> doc;
    doc["device_id"]    = DEVICE_ID;
    doc["uptime_ms"]    = millis();
    doc["free_heap"]    = ESP.getFreeHeap();
    doc["wifi_rssi"]    = WiFi.RSSI();
    doc["count"]        = verificationCount;
    doc["firmware"]     = FIRMWARE_VERSION;
    
    String payload;
    serializeJson(doc, payload);
    
    mqttClient.publish(TOPIC_STATUS, payload.c_str(), false);
    Serial.printf("[INFO] Heartbeat: heap=%d, rssi=%d\n",
                  ESP.getFreeHeap(), WiFi.RSSI());
}

// ============================================================
// Session Management
// ============================================================

bool initFingerprintSensor() {
    fingerprintSerial.begin(57600, SERIAL_8N1,
                            FINGERPRINT_RX_PIN, FINGERPRINT_TX_PIN);
    
    if (fingerprintSensor.verifyPassword()) {
        Serial.println("[INFO] R307 sensor ready");
        fingerprintSensor.getParameters();
        Serial.printf("[INFO] Sensor capacity: %d templates\n",
                      fingerprintSensor.capacity);
        return true;
    }
    
    Serial.println("[ERROR] R307 sensor password verification failed");
    return false;
}

void resetSession() {
    lastResult = {
        .decided = false,
        .decision = 0,
        .belief = 0.0f,
        .similarity_score = 0.0f,
        .liveness_score = 0.0f,
        .reason = "",
        .audit_log_id = "",
        .latency_ms = 0
    };
    
    indicateLED(LED_GREEN_PIN, false);
    indicateLED(LED_RED_PIN, false);
}
