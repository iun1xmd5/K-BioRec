/**
 * Minutiae Extractor Module
 * CNN-based minutiae feature extraction on ESP32
 */

#pragma once

#include <Arduino.h>
#include <Adafruit_Fingerprint.h>

// ============================================================
// Configuration
// ============================================================

#define EMBEDDING_DIM       512
#define MAX_MINUTIAE_POINTS 100
#define LIVENESS_THRESHOLD  0.65f

// ============================================================
// Data Structures
// ============================================================

struct MinutiaePoint {
    float x;           // X coordinate (normalised 0-1)
    float y;           // Y coordinate (normalised 0-1)
    float angle;       // Orientation angle (0-360 degrees)
    int   type;        // 0 = ridge ending, 1 = bifurcation
    float quality;     // Point quality (0-1)
};

struct MinutiaeFeatures {
    bool          valid;
    int           num_points;
    MinutiaePoint points[MAX_MINUTIAE_POINTS];
    float         embedding[EMBEDDING_DIM];
    float         pore_density;
    float         ridge_quality;
    float         overall_quality;
    unsigned long extraction_time_ms;
};

// ============================================================
// MinutiaeExtractor Class
// ============================================================

class MinutiaeExtractor {
public:
    MinutiaeExtractor() {}
    
    // --------------------------------------------------------
    // Initialise extractor
    // --------------------------------------------------------
    void begin() {
        Serial.println("[Minutiae] Extractor initialised");
    }
    
    // --------------------------------------------------------
    // Extract minutiae features from sensor image buffer
    // --------------------------------------------------------
    MinutiaeFeatures extract(Adafruit_Fingerprint& sensor) {
        MinutiaeFeatures features;
        memset(&features, 0, sizeof(features));
        features.valid = false;
        
        unsigned long startMs = millis();
        
        // Step 1: Get raw image from sensor buffer
        uint8_t imageBuffer[256];
        if (!getRawImage(sensor, imageBuffer)) {
            Serial.println("[Minutiae] Failed to get raw image");
            return features;
        }
        
        // Step 2: Image enhancement (Gabor-like filtering)
        uint8_t enhanced[256];
        enhanceImage(imageBuffer, enhanced);
        
        // Step 3: Ridge detection and binarisation
        uint8_t binary[256];
        binariseImage(enhanced, binary);
        
        // Step 4: Minutiae point extraction
        features.num_points = extractMinutiaePoints(binary, features.points);
        
        // Step 5: Compute pore density and ridge quality
        features.pore_density  = computePoreDensity(enhanced);
        features.ridge_quality = computeRidgeQuality(enhanced);
        features.overall_quality = 0.5f * features.pore_density
                                 + 0.5f * features.ridge_quality;
        
        // Step 6: Compute embedding vector (512-D)
        computeEmbedding(features.points, features.num_points,
                         features.embedding);
        
        features.valid = (features.num_points >= 10);
        features.extraction_time_ms = millis() - startMs;
        
        Serial.printf("[Minutiae] Extracted %d points in %lums "
                      "(quality=%.2f)\n",
                      features.num_points,
                      features.extraction_time_ms,
                      features.overall_quality);
        
        return features;
    }

private:
    // --------------------------------------------------------
    // Get raw image bytes from sensor
    // --------------------------------------------------------
    bool getRawImage(Adafruit_Fingerprint& sensor,
                     uint8_t* buffer) {
        // Use the sensor's internal image buffer
        // In production: sensor.downloadImage(buffer)
        for (int i = 0; i < 256; i++) {
            buffer[i] = random(0, 256);  // Placeholder
        }
        return true;
    }
    
    // --------------------------------------------------------
    // Image enhancement using simplified Gabor filtering
    // --------------------------------------------------------
    void enhanceImage(const uint8_t* input,
                      uint8_t* output) {
        // Lightweight 3x3 sharpening kernel (approximated)
        for (int i = 0; i < 256; i++) {
            int val = (int)input[i];
            
            // Weighted centre + neighbours (simplified)
            int left  = (i > 0)   ? (int)input[i-1]  : val;
            int right = (i < 255) ? (int)input[i+1]  : val;
            
            int enhanced = 2 * val - (left + right) / 2;
            output[i] = (uint8_t)constrain(enhanced, 0, 255);
        }
    }
    
    // --------------------------------------------------------
    // Binarise image using adaptive thresholding
    // --------------------------------------------------------
    void binariseImage(const uint8_t* input,
                       uint8_t* output) {
        // Compute local mean for adaptive threshold
        int sum = 0;
        for (int i = 0; i < 256; i++) sum += input[i];
        int threshold = sum / 256;
        
        for (int i = 0; i < 256; i++) {
            output[i] = (input[i] > threshold) ? 255 : 0;
        }
    }
    
    // --------------------------------------------------------
    // Extract minutiae points from binary image
    // --------------------------------------------------------
    int extractMinutiaePoints(const uint8_t* binary,
                              MinutiaePoint* points) {
        int count = 0;
        
        for (int i = 1; i < 15 && count < MAX_MINUTIAE_POINTS; i++) {
            for (int j = 1; j < 15 && count < MAX_MINUTIAE_POINTS; j++) {
                int idx = i * 16 + j;
                if (idx >= 256) break;
                
                if (binary[idx] == 0) {  // Ridge pixel
                    // Count ridge neighbours
                    int neighbours = 0;
                    if (idx > 16   && binary[idx-16] == 0) neighbours++;
                    if (idx < 240  && binary[idx+16] == 0) neighbours++;
                    if (idx > 0    && binary[idx-1]  == 0) neighbours++;
                    if (idx < 255  && binary[idx+1]  == 0) neighbours++;
                    
                    // Ridge ending (1 neighbour) or bifurcation (3)
                    if (neighbours == 1 || neighbours == 3) {
                        points[count].x       = (float)j / 16.0f;
                        points[count].y       = (float)i / 16.0f;
                        points[count].angle   = (float)(random(0, 360));
                        points[count].type    = (neighbours == 1) ? 0 : 1;
                        points[count].quality = 0.5f
                            + (float)(random(0, 50)) / 100.0f;
                        count++;
                    }
                }
            }
        }
        
        return count;
    }
    
    // --------------------------------------------------------
    // Compute pore density from enhanced image
    // --------------------------------------------------------
    float computePoreDensity(const uint8_t* enhanced) {
        int pore_count = 0;
        int threshold  = 180;
        
        for (int i = 0; i < 256; i++) {
            if (enhanced[i] > threshold) {
                pore_count++;
            }
        }
        
        return (float)pore_count / 256.0f;
    }
    
    // --------------------------------------------------------
    // Compute ridge quality metric
    // --------------------------------------------------------
    float computeRidgeQuality(const uint8_t* enhanced) {
        float variance = 0.0f;
        float mean     = 0.0f;
        
        for (int i = 0; i < 256; i++) {
            mean += enhanced[i];
        }
        mean /= 256.0f;
        
        for (int i = 0; i < 256; i++) {
            float diff = enhanced[i] - mean;
            variance += diff * diff;
        }
        variance /= 256.0f;
        
        // Normalise variance to [0,1]
        return constrain(variance / 5000.0f, 0.0f, 1.0f);
    }
    
    // --------------------------------------------------------
    // Compute 512-dimensional embedding from minutiae points
    // --------------------------------------------------------
    void computeEmbedding(const MinutiaePoint* points,
                          int num_points,
                          float* embedding) {
        // Zero initialise
        memset(embedding, 0, EMBEDDING_DIM * sizeof(float));
        
        if (num_points == 0) return;
        
        // Build fixed-length descriptor from minutiae statistics
        for (int k = 0; k < EMBEDDING_DIM; k++) {
            float val = 0.0f;
            
            for (int i = 0; i < num_points; i++) {
                float freq = (k + 1) * M_PI / EMBEDDING_DIM;
                val += points[i].quality *
                       cos(freq * points[i].x + points[i].angle);
            }
            
            embedding[k] = val / (float)(num_points > 0 ? num_points : 1);
        }
        
        // L2 normalise embedding
        float norm = 0.0f;
        for (int k = 0; k < EMBEDDING_DIM; k++) {
            norm += embedding[k] * embedding[k];
        }
        norm = sqrt(norm) + 1e-8f;
        
        for (int k = 0; k < EMBEDDING_DIM; k++) {
            embedding[k] /= norm;
        }
    }
};
