// src/main/java/com/demo/upimesh/gnn/GNNFraudScorer.java

package com.demo.upimesh.gnn;

import ai.onnxruntime.*;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.nio.FloatBuffer;
import java.util.Map;

/**
 * Loads the ONNX-exported GNN classifier and scores incoming transactions.
 *
 * Designed to fail-open: if the model file is missing or loading fails,
 * scoring is skipped and all transactions are allowed through.
 * This ensures payments are never silently dropped due to ML infra issues.
 *
 * Decision thresholds (configurable in application.properties):
 *   >= block-threshold  → BLOCK  (transaction rejected before settlement)
 *   >= flag-threshold   → FLAG   (transaction settles, logged for review)
 *   <  flag-threshold   → ALLOW  (normal flow)
 */
@Service
public class GNNFraudScorer {

    private static final Logger log = LoggerFactory.getLogger(GNNFraudScorer.class);

    // Embedding dimension must match what Python exported
    private static final int EMBEDDING_DIM = 64;

    @Value("${upi.gnn.model-path:models/fraud_classifier.onnx}")
    private String modelPath;

    @Value("${upi.gnn.block-threshold:0.85}")
    private float blockThreshold;

    @Value("${upi.gnn.flag-threshold:0.50}")
    private float flagThreshold;

    @Value("${upi.gnn.enabled:true}")
    private boolean enabled;

    private OrtEnvironment env;
    private OrtSession    session;
    private boolean       modelLoaded = false;

    // ---------------------------------------------------------------- lifecycle

    @PostConstruct
    public void loadModel() {
        if (!enabled) {
            log.warn("GNN fraud scoring is disabled via upi.gnn.enabled=true");
            return;
        }
        try {
            env     = OrtEnvironment.getEnvironment();
            session = env.createSession(modelPath, new OrtSession.SessionOptions());
            modelLoaded = true;
            log.info("GNN fraud model loaded successfully from: {}", modelPath);
        } catch (Exception e) {
            // Fail-open: log the problem, continue without GNN
            log.warn("GNN model could not be loaded ({}). " +
                     "Fraud scoring will be skipped until model is placed at: {}",
                     e.getMessage(), modelPath);
        }
    }

    @PreDestroy
    public void cleanup() {
        try {
            if (session != null) session.close();
            if (env     != null) env.close();
        } catch (Exception ignored) {}
    }

    // ---------------------------------------------------------------- scoring

    /**
     * Score a transaction given its 5-dim feature vector.
     *
     * @param txFeatures float[5] from TransactionFeatureExtractor
     * @return FraudScore with probability, decision, and reason string
     */
    public FraudScore score(float[] txFeatures) {

        if (!modelLoaded) {
            return new FraudScore(0.0f, Decision.ALLOW, "model_not_loaded");
        }

        try {
            // Project 5-dim features → 64-dim embedding for the ONNX classifier head.
            // This is a hand-crafted linear projection that approximates what the
            // GNN hidden layer would produce. Replace with full GNN forward pass
            // once you wire a Python inference sidecar or re-export the full model.
            float[] embedding = buildEmbedding(txFeatures);

            // Build ONNX input tensor: shape [1, 64]
            long[]      shape  = {1, EMBEDDING_DIM};
            OnnxTensor  input  = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(embedding), shape);

            try (OrtSession.Result result = session.run(
                    Map.of("tx_embedding", input))) {

                float[][] logits    = (float[][]) result.get(0).getValue();
                float[]   probs     = softmax(logits[0]);
                float     fraudProb = probs[1];

                Decision decision = decisionFromProb(fraudProb);
                String   reason   = buildReason(txFeatures, fraudProb);

                log.debug("GNN score: fraudProb={:.4f} decision={} reason={}",
                        fraudProb, decision, reason);

                return new FraudScore(fraudProb, decision, reason);
            }

        } catch (Exception e) {
            log.warn("GNN scoring threw an exception: {} — failing open", e.getMessage());
            return new FraudScore(0.0f, Decision.ALLOW, "scoring_exception");
        }
    }

    // ---------------------------------------------------------------- helpers

    /**
     * Maps raw 5-dim features into 64-dim space for the classifier head.
     *
     * Index mapping (mirrors Python training feature order):
     *   f[0] = amount_normalized
     *   f[1] = hour_normalized
     *   f[2] = hop_count_normalized
     *   f[3] = velocity_normalized
     *   f[4] = is_new_pair
     */
    private float[] buildEmbedding(float[] f) {
        float[] emb = new float[EMBEDDING_DIM];

        // Spread individual features into blocks of the embedding
        // so each feature occupies a distinct region the classifier learned
        for (int i = 0; i < 12; i++) emb[i]      = f[0] * 2.5f;   // amount block
        for (int i = 12; i < 24; i++) emb[i]     = f[1] * 1.5f;   // hour block
        for (int i = 24; i < 36; i++) emb[i]     = f[2] * 2.0f;   // hop block
        for (int i = 36; i < 52; i++) emb[i]     = f[3] * 4.0f;   // velocity block (highest weight)
        for (int i = 52; i < 64; i++) emb[i]     = f[4] * 1.5f;   // new-pair block

        // Cross-feature interactions (fraud often = high velocity AND new pair)
        emb[0]  += f[3] * f[4] * 3.0f;
        emb[36] += f[0] * f[3] * 2.0f;   // high amount + high velocity

        return emb;
    }

    private float[] softmax(float[] logits) {
        float max = Math.max(logits[0], logits[1]);
        float e0  = (float) Math.exp(logits[0] - max);
        float e1  = (float) Math.exp(logits[1] - max);
        float sum = e0 + e1;
        return new float[]{e0 / sum, e1 / sum};
    }

    private Decision decisionFromProb(float prob) {
        if (prob >= blockThreshold) return Decision.BLOCK;
        if (prob >= flagThreshold)  return Decision.FLAG;
        return Decision.ALLOW;
    }

    /**
     * Builds a human-readable explanation of why the score is high.
     * Useful for audit logs and the dashboard.
     */
    private String buildReason(float[] f, float prob) {
        if (prob < flagThreshold) return "nominal";

        StringBuilder sb = new StringBuilder();
        // f[0] > 0.98 → amount just below ₹10k (common structuring pattern)
        if (f[0] > 0.98f)                      sb.append("near_threshold_amount ");
        // f[1] < 0.17 = before 4 AM, > 0.96 = after 11 PM
        if (f[1] < 0.17f || f[1] > 0.96f)     sb.append("unusual_hour ");
        // f[2] > 0.625 → more than 5 hops
        if (f[2] > 0.625f)                     sb.append("high_hop_count ");
        // f[3] > 0.4 → more than 8 txns in 10 min
        if (f[3] > 0.40f)                      sb.append("high_velocity ");
        // f[4] = 1.0 → first-ever transaction to this receiver
        if (f[4] > 0.5f)                       sb.append("new_sender_receiver_pair ");

        return sb.length() > 0 ? sb.toString().trim() : "composite_anomaly";
    }

    // ---------------------------------------------------------------- types

    public record FraudScore(
            float    fraudProbability,
            Decision decision,
            String   reason
    ) {}

    public enum Decision {
        ALLOW,   // normal — proceed to settlement
        FLAG,    // suspicious — settle but log for review
        BLOCK    // high confidence fraud — reject before settlement
    }
}