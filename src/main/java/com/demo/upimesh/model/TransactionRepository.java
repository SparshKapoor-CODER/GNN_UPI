package com.demo.upimesh.model;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.Instant;
import java.util.List;

public interface TransactionRepository extends JpaRepository<Transaction, Long> {

    // ── existing methods (unchanged) ──────────────────────────────────────────

    List<Transaction> findTop20ByOrderByIdDesc();

    boolean existsByPacketHash(String packetHash);

    // ── added for GNN feature extraction (Step 4.1) ───────────────────────────

    /**
     * Counts how many transactions a sender successfully settled
     * within a given time window.
     *
     * Used by TransactionFeatureExtractor to compute the
     * "velocity" feature [3] = txns in last 10 minutes.
     *
     * High velocity (many txns in a short window from one sender)
     * is one of the strongest fraud signals in the GNN model.
     */
    @Query("SELECT COUNT(t) FROM Transaction t " +
           "WHERE t.senderVpa = :sender " +
           "AND t.settledAt >= :from " +
           "AND t.settledAt < :to " +
           "AND t.status = 'SETTLED'")
    long countRecentBySender(@Param("sender") String senderVpa,
                             @Param("from")   Instant from,
                             @Param("to")     Instant to);

    /**
     * Returns true if this exact sender→receiver pair has
     * at least one prior settled transaction in the ledger.
     *
     * Used by TransactionFeatureExtractor to compute the
     * "is_new_pair" feature [4].
     *
     * A large amount sent to a receiver the sender has never
     * transacted with before is a strong fraud indicator.
     */
    boolean existsBySenderVpaAndReceiverVpa(String senderVpa,
                                            String receiverVpa);
}