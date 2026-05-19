// src/main/java/com/demo/upimesh/gnn/TransactionFeatureExtractor.java

package com.demo.upimesh.gnn;

import com.demo.upimesh.model.PaymentInstruction;
import com.demo.upimesh.model.TransactionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.ZoneId;

/**
 * Converts a live PaymentInstruction into the same 5-dim float vector
 * that was used during Python training.
 *
 * ORDER MUST MATCH build_graph.py tx_features exactly:
 *   [0] amount_normalized      = amount / 10000   (capped at 1.0)
 *   [1] hour_of_day_normalized = hour  / 24.0
 *   [2] hop_count_normalized   = hops  / 8.0      (capped at 1.0)
 *   [3] velocity_normalized    = txns_last_10min / 20.0
 *   [4] is_new_pair            = 1.0 if no prior history, else 0.0
 */
@Component
public class TransactionFeatureExtractor {

    @Autowired
    private TransactionRepository txRepo;

    /**
     * @param instruction  the decrypted payment payload
     * @param hopCount     how many mesh hops the packet travelled
     * @return float[5] feature vector, values in [0, 1]
     */
    public float[] extract(PaymentInstruction instruction, int hopCount) {

        float[] f = new float[5];

        // [0] Amount — normalize to ₹10,000 ceiling
        double amount = instruction.getAmount().doubleValue();
        f[0] = (float) Math.min(amount / 10_000.0, 1.0);

        // [1] Hour of day in IST
        int hour = Instant.ofEpochMilli(instruction.getSignedAt())
                .atZone(ZoneId.of("Asia/Kolkata"))
                .getHour();
        f[1] = hour / 24.0f;

        // [2] Hop count — more hops = more intermediaries = higher risk
        f[2] = Math.min(hopCount / 8.0f, 1.0f);

        // [3] Velocity — how many settled txns did this sender
        //     make in the 10-minute window ending at signedAt?
        Instant signedAt  = Instant.ofEpochMilli(instruction.getSignedAt());
        Instant tenMinAgo = signedAt.minusSeconds(600);
        long recentCount  = txRepo.countRecentBySender(
                instruction.getSenderVpa(), tenMinAgo, signedAt);
        f[3] = Math.min(recentCount / 20.0f, 1.0f);

        // [4] New pair — sender has never successfully sent to this receiver
        boolean hasPriorHistory = txRepo.existsBySenderVpaAndReceiverVpa(
                instruction.getSenderVpa(),
                instruction.getReceiverVpa());
        f[4] = hasPriorHistory ? 0.0f : 1.0f;

        return f;
    }
}