//!/usr/bin/env python3
 //-*- coding: utf-8 -*-

//Created on Thu Apr 16 22:09:29 2026
//@author: dr

/**
 * Communication Module
 * MQTT/TLS secure messaging and payload management
 */

#pragma once

#include <Arduino.h>
#include <ArduinoJson.h>
#include <WiFiClientSecure.h>
#include <mbedtls/aes.h>
#include <mbedtls/base64.h>
#include "minutiae_extractor.hpp"

// ============================================================
// Configuration
// ============================================================

#define SESSION_ID_LEN    16
#define AES_KEY_SIZE      32   // 256-bit AES
#define MAX_PAYLOAD_SIZE  4096

// ============================================================
// Data Structures
// ============================================================

struct ProbePayload {
    String session_id;
    String candidate_id;
    float  liveness_score;
    String encrypted_embedding;   // Base64-encoded AES-256
    String device_id;
    unsigned long timestamp_ms;
};

// ============================================================
// Communication Class
// ============================================================

class Communication {
public:
    Communication() {
        memset(aes_key, 0, AES_KEY_SIZE);
        candidate_id    = "UNKNOWN";
        liveness_score  = 0.0f;
        session_id      = "";
    }
    
    // --------------------------------------------------------
    // Initialise communication module
    // --------------------------------------------------------
    void begin(const uint8_t* key) {
        memcpy(aes_key, key, AES_KEY_SIZE);
        generateSessionId();
        Serial.println("[Comm] Communication module initialised");
        Serial.println("[Comm] Session ID: " + session_id);
    }
    
    // --------------------------------------------------------
    // Set probe features for transmission
    // --------------------------------------------------------
    void setProbeFeatures(const MinutiaeFeatures& features,
                          float liveness) {
        liveness_score = liveness;
        
        // Encrypt embedding using AES-256
        encrypted_embedding = encryptEmbedding(
            features.embedding, EMBEDDING_DIM
        );
    }
    
    // --------------------------------------------------------
    // Build probe JSON payload
    // --------------------------------------------------------
    String buildPayload(const char* device_id) {
        StaticJsonDocument<2048> doc;
        
        doc["session_id"]              = session_id;
        doc["candidate_id"]            = candidate_id;
        doc["device_id"]               = device_id;
        doc["liveness_score"]          = liveness_score;
        doc["fingerprint_embedding"]   = encrypted_embedding;
        doc["timestamp_ms"]            = millis();
        doc["protocol_version"]        = "1.0";
        
        String payload;
        serializeJson(doc, payload);
        return payload;
    }
    
    // --------------------------------------------------------
    // Getters
    // --------------------------------------------------------
    String getCandidateId()         { return candidate_id; }
    float  getLivenessScore()       { return liveness_score; }
    String getSessionId()           { return session_id; }
    String getEncryptedEmbedding()  { return encrypted_embedding; }
    
    // --------------------------------------------------------
    // Setters
    // --------------------------------------------------------
    void setCandidateId(const String& id) {
        candidate_id = id;
        Serial.println("[Comm] Candidate ID set: " + id);
    }
    
    // --------------------------------------------------------
    // Rotate session ID (call after each verification)
    // --------------------------------------------------------
    void rotateSession() {
        generateSessionId();
        Serial.println("[Comm] Session rotated: " + session_id);
    }

private:
    uint8_t aes_key[AES_KEY_SIZE];
    String  session_id;
    String  candidate_id;
    float   liveness_score;
    String  encrypted_embedding;
    
    // --------------------------------------------------------
    // Generate cryptographically random session ID
    // --------------------------------------------------------
    void generateSessionId() {
        session_id = "";
        
        for (int i = 0; i < SESSION_ID_LEN; i++) {
            char c;
            int r = random(0, 36);
            if (r < 10) c = '0' + r;
            else        c = 'a' + (r - 10);
            session_id += c;
        }
    }
    
    // --------------------------------------------------------
    // Encrypt embedding using AES-256-ECB
    // Returns Base64-encoded ciphertext
    // --------------------------------------------------------
    String encryptEmbedding(const float* embedding, int dim) {
        // Convert float array to bytes
        size_t byte_len = dim * sizeof(float);
        uint8_t* input  = (uint8_t*)embedding;
        
        // Pad to AES block boundary (16 bytes)
        size_t padded_len = ((byte_len + 15) / 16) * 16;
        uint8_t padded[padded_len];
        memset(padded, 0, padded_len);
        memcpy(padded, input, byte_len);
        
        // AES-256-ECB encryption via mbedTLS
        mbedtls_aes_context aes;
        mbedtls_aes_init(&aes);
        mbedtls_aes_setkey_enc(&aes, aes_key, 256);
        
        uint8_t ciphertext[padded_len];
        
        for (size_t i = 0; i < padded_len; i += 16) {
            mbedtls_aes_crypt_ecb(
                &aes,
                MBEDTLS_AES_ENCRYPT,
                &padded[i],
                &ciphertext[i]
            );
        }
        
        mbedtls_aes_free(&aes);
        
        // Base64 encode
        size_t b64_len = 0;
        mbedtls_base64_encode(nullptr, 0, &b64_len,
                              ciphertext, padded_len);
        
        uint8_t b64_buffer[b64_len + 1];
        mbedtls_base64_encode(b64_buffer, b64_len + 1, &b64_len,
                              ciphertext, padded_len);
        b64_buffer[b64_len] = '\0';
        
        return String((char*)b64_buffer);
    }
};
