/**
 * Fuzzy Liveness Detection Module
 * Pore density + ridge quality fusion for spoof rejection
 * 
 * Implements: b_l = 0.4 * p_o + 0.6 * r_q
 * Optimal weights derived from LivDet 2021 ablation study
 */

#pragma once

#include <Arduino.h>
#include "minutiae_extractor.hpp"

// ============================================================
// Configuration
// ============================================================

#define LIVENESS_THRESHOLD      0.65f   // Optimal gate threshold
#define PORE_WEIGHT             0.4f    // Pore density weight
#define RIDGE_WEIGHT            0.6f    // Ridge quality weight

// ============================================================
// Data Structures
// ============================================================

struct LivenessResult {
    float score;          // Final liveness belief [0,1]
    float pore_score;     // Pore density membership
    float ridge_score;    // Ridge quality membership
    bool  is_live;        // True if score >= threshold
    String decision_reason;
};

struct FuzzyMembership {
    float low;
    float medium;
    float high;
};

// ============================================================
// FuzzyLiveness Class
// ============================================================

class FuzzyLiveness {
public:
    // Configurable weights
    float pore_weight  = PORE_WEIGHT;
    float ridge_weight = RIDGE_WEIGHT;
    float threshold    = LIVENESS_THRESHOLD;
    
    FuzzyLiveness() {}
    
    // --------------------------------------------------------
    // Initialise fuzzy liveness module
    // --------------------------------------------------------
    void begin() {
        Serial.printf("[Liveness] Fuzzy liveness initialised "
                      "(pore_w=%.1f, ridge_w=%.1f, thresh=%.2f)\n",
                      pore_weight, ridge_weight, threshold);
    }
    
    // --------------------------------------------------------
    // Main liveness assessment
    // b_l = pore_weight * p_o + ridge_weight * r_q
    // --------------------------------------------------------
    LivenessResult assess(const MinutiaeFeatures& features) {
        LivenessResult result;
        
        // Step 1: Fuzzify pore density
        FuzzyMembership pore_membership =
            fuzzify(features.pore_density);
        
        // Step 2: Fuzzify ridge quality
        FuzzyMembership ridge_membership =
            fuzzify(features.ridge_quality);
        
        // Step 3: Apply fuzzy rules
        float pore_score  = defuzzify(pore_membership);
        float ridge_score = defuzzify(ridge_membership);
        
        // Step 4: Weighted fusion
        // b_l = 0.4 * p_o + 0.6 * r_q
        float liveness_score = pore_weight  * pore_score
                             + ridge_weight * ridge_score;
        
        // Step 5: Clip to valid range
        liveness_score = constrain(liveness_score, 0.0f, 1.0f);
        
        // Step 6: Decision
        bool is_live = (liveness_score >= threshold);
        
        result.score   = liveness_score;
        result.pore_score  = pore_score;
        result.ridge_score = ridge_score;
        result.is_live = is_live;
        result.decision_reason = buildReason(
            pore_score, ridge_score,
            liveness_score, is_live
        );
        
        Serial.printf(
            "[Liveness] p_o=%.3f, r_q=%.3f, "
            "b_l=%.3f => %s\n",
            pore_score, ridge_score, liveness_score,
            is_live ? "LIVE" : "SPOOF"
        );
        
        return result;
    }
    
    // --------------------------------------------------------
    // Update weights at runtime
    // --------------------------------------------------------
    void setWeights(float pore_w, float ridge_w) {
        if (abs(pore_w + ridge_w - 1.0f) < 1e-3f) {
            pore_weight  = pore_w;
            ridge_weight = ridge_w;
            Serial.printf("[Liveness] Weights updated: "
                          "pore=%.2f, ridge=%.2f\n",
                          pore_weight, ridge_weight);
        } else {
            Serial.println("[Liveness] Weights must sum to 1.0");
        }
    }

private:
    // --------------------------------------------------------
    // Fuzzify a crisp value into low/medium/high membership
    // Triangular membership functions
    // --------------------------------------------------------
    FuzzyMembership fuzzify(float value) {
        FuzzyMembership m;
        
        // Low:    trimf(0.0, 0.0, 0.4)
        m.low = triangleMF(value, 0.0f, 0.0f, 0.4f);
        
        // Medium: trimf(0.2, 0.5, 0.8)
        m.medium = triangleMF(value, 0.2f, 0.5f, 0.8f);
        
        // High:   trimf(0.6, 1.0, 1.0)
        m.high = triangleMF(value, 0.6f, 1.0f, 1.0f);
        
        return m;
    }
    
    // --------------------------------------------------------
    // Triangular membership function
    // --------------------------------------------------------
    float triangleMF(float x, float a, float b, float c) {
        if (x <= a || x >= c) return 0.0f;
        if (x == b)           return 1.0f;
        if (x < b) return (x - a) / (b - a + 1e-8f);
        return (c - x) / (c - b + 1e-8f);
    }
    
    // --------------------------------------------------------
    // Defuzzify using centroid method
    // Maps membership to crisp score [0,1]
    // --------------------------------------------------------
    float defuzzify(const FuzzyMembership& m) {
        // Centroid of output membership
        // Low → 0.15, Medium → 0.50, High → 0.85
        float numerator   = m.low * 0.15f
                          + m.medium * 0.50f
                          + m.high * 0.85f;
        float denominator = m.low + m.medium + m.high + 1e-8f;
        
        return numerator / denominator;
    }
    
    // --------------------------------------------------------
    // Build human-readable decision reason
    // --------------------------------------------------------
    String buildReason(float pore, float ridge,
                       float score, bool is_live) {
        String reason = "";
        
        if (!is_live) {
            if (pore < 0.3f)
                reason += "Low pore density (possible silicone). ";
            if (ridge < 0.3f)
                reason += "Poor ridge quality (possible gelatin). ";
            if (score < 0.5f)
                reason += "Critically low liveness score. ";
        } else {
            reason = "Liveness confirmed (score=" +
                     String(score, 3) + ")";
        }
        
        return reason;
    }
};
