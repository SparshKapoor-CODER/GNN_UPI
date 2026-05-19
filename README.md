# UPI Offline Mesh + GNN Fraud Detection

A Spring Boot backend demonstrating **offline UPI payments routed through a Bluetooth-style mesh network**, now with a **Graph Neural Network (GNN) fraud detection layer** that scores every transaction before settlement.

> You're in a basement with zero connectivity. You send your friend ₹500. Your phone encrypts the payment, broadcasts it to nearby phones, and the packet hops device-to-device until *some* phone walks outside, gets 4G, and silently uploads it to this backend. The backend decrypts, deduplicates, **scores it with a GNN**, and settles.

---

## Table of Contents

1. [What this demo proves](#what-this-demo-proves)
2. [How to run it](#how-to-run-it)
3. [The demo flow](#the-demo-flow-step-by-step)
4. [Architecture](#architecture)
5. [The four hard problems and how they're solved](#the-four-hard-problems-and-how-theyre-solved)
6. [GNN Fraud Detection](#gnn-fraud-detection)
7. [File-by-file walkthrough](#file-by-file-walkthrough)
8. [API reference](#api-reference)
9. [Tests](#tests)
10. [What's NOT real](#whats-not-real-and-what-would-change-for-production)
11. [Honest limitations](#honest-limitations-of-the-concept)
12. [Contributors](#contributors)

---

## What this demo proves

1. **A payment can travel from sender to backend through untrusted intermediaries** without any of them being able to read or tamper with it. (Hybrid RSA + AES-GCM encryption.)
2. **Even if the same payment reaches the backend simultaneously through multiple bridge nodes, it settles exactly once.** (Idempotency via atomic compare-and-set on the ciphertext hash.)
3. **A tampered or replayed packet is rejected** before it touches the ledger.
4. **A Graph Neural Network scores every transaction** for fraud before settlement — blocking suspicious payments and flagging borderline ones for review, without blocking legitimate ones.

---

## How to run it

### Prerequisites

- **JDK 17 or newer** on PATH (or `JAVA_HOME` set). Check with `java -version`.
- The trained ONNX model at `models/fraud_classifier.onnx` (already included).
- That's it. No database, no Redis, no separate ML server.

### Windows

```cmd
mvnw.cmd spring-boot:run
```

### Mac / Linux

```bash
./mvnw spring-boot:run
```

First run downloads Maven + dependencies (~80 MB). Subsequent runs start in ~10 seconds.

### Open the dashboard

Once you see `Started UpiMeshApplication`, open:

**http://localhost:8080**

You'll get a dark dashboard with live GNN scores, fraud stats, and the full mesh simulation.

---

## The demo flow (step by step)

### Step 1 — Compose a payment

Choose sender, receiver, amount, PIN → **📤 Inject into Mesh**.

The server simulates the sender's phone: builds a `PaymentInstruction`, encrypts it with RSA+AES, wraps it in a `MeshPacket` with TTL=5, hands it to `phone-alice`.

### Step 2 — Gossip

Click **🔄 Run Gossip Round** twice.

Each round, every device broadcasts its packets to every nearby device. TTL decrements each hop. After 2 rounds every device holds the packet — including `phone-bridge`.

### Step 3 — Bridge node walks outside

Click **📡 Bridges Upload to Backend**.

`phone-bridge` (the only device with `hasInternet=true`) POSTs all its packets to `/api/bridge/ingest`. The full pipeline runs per packet:

1. Hash the ciphertext (SHA-256)
2. Idempotency gate — duplicate? Drop.
3. Decrypt with the server's RSA private key
4. Freshness check — stale/replayed? Reject.
5. **GNN fraud scoring** — BLOCK / FLAG / ALLOW
6. Settlement — debit sender, credit receiver, write ledger

### Step 4 — Watch the GNN

The **Last Bridge Upload Results** panel shows a score pill per upload:

| Pill | Meaning |
|---|---|
| `✅ 2.3%` | Low fraud probability — ALLOWED, proceeds to settlement |
| `⚠️ 61.4%` | Medium — FLAGGED, settles but logged for review |
| `🚫 91.2%` | High — BLOCKED, transaction rejected before ledger |

The **Transaction Ledger** also shows `GNN Score` and `Reason` columns pulled directly from the database.

### Step 5 — Test a fraud scenario

Send `alice@demo → dave@demo` with amount `9999` (near-threshold structuring pattern + new sender-receiver pair). After flush, watch the log:

```
GNN score for alice@demo→dave@demo ₹9999.00: prob=71.4% decision=FLAG
reason=near_threshold_amount new_sender_receiver_pair
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        SENDER PHONE (offline)                            │
│  PaymentInstruction { sender, receiver, amount, pinHash, nonce, time }   │
│              │                                                           │
│              ▼  encrypt with server's RSA public key (hybrid RSA+AES)   │
│   MeshPacket { packetId, ttl, createdAt, ciphertext }                    │
└─────────────────────────────────────┬────────────────────────────────────┘
                                      │  Bluetooth gossip (BLE / Wi-Fi Direct)
                                      ▼
       ┌──────────┐  hop  ┌──────────┐  hop  ┌──────────┐
       │stranger1 │ ────▶ │stranger2 │ ────▶ │  bridge  │ ◀── walks outside
       └──────────┘       └──────────┘       └────┬─────┘     gets 4G
                                                  │
                                                  ▼  HTTPS POST
┌──────────────────────────────────────────────────────────────────────────┐
│                    SPRING BOOT BACKEND (this project)                    │
│                                                                          │
│  /api/bridge/ingest                                                      │
│       │                                                                  │
│       ▼  [1] SHA-256(ciphertext)                                         │
│       │                                                                  │
│       ▼  [2] IdempotencyService.claim(hash)   ← ConcurrentHashMap SETNX │
│       │       duplicate? → DUPLICATE_DROPPED                             │
│       │                                                                  │
│       ▼  [3] HybridCryptoService.decrypt()    ← RSA-OAEP + AES-256-GCM  │
│       │       tampered?  → INVALID                                       │
│       │                                                                  │
│       ▼  [4] Freshness check (signedAt window)                           │
│       │       stale/future? → INVALID                                    │
│       │                                                                  │
│       ▼  [5] GNNFraudScorer.score(features)   ← ONNX Runtime            │
│       │       BLOCK (≥0.85)? → BLOCKED_BY_GNN                           │
│       │       FLAG  (≥0.50)? → settle + log                              │
│       │                                                                  │
│       ▼  [6] SettlementService.settle()        ← @Transactional          │
│               debit sender, credit receiver, write ledger + GNN metadata │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## The four hard problems and how they're solved

### Problem 1: Untrusted intermediates

A stranger's phone carries your transaction. How do you prevent them reading the amount or modifying it?

**Solution: Hybrid encryption (RSA-OAEP + AES-GCM).**

1. Generate a one-time AES-256 key per packet.
2. Encrypt the payment JSON with **AES-256-GCM** (fast + authenticated).
3. Encrypt the AES key with **RSA-OAEP** using the server's public key.
4. Wire format: `[256-byte RSA-encrypted AES key][12-byte IV][AES ciphertext + 16-byte GCM tag]`

AES-GCM is *authenticated* encryption — a single flipped bit anywhere throws an exception on decryption. Intermediates cannot tamper undetected. See `HybridCryptoService.java`.

### Problem 2: The duplicate-storm

Three bridges hold the same packet, walk outside simultaneously, and POST to the backend within milliseconds. Naive processing debits the sender three times.

**Solution: Atomic compare-and-set on the ciphertext hash.**

```java
// IdempotencyService.java
Instant prev = seen.putIfAbsent(packetHash, now);
return prev == null;  // true = first claimer, false = duplicate
```

`ConcurrentHashMap.putIfAbsent` is atomic at the JVM level. Exactly one thread gets `null` back; the rest are short-circuited as `DUPLICATE_DROPPED` before any decryption or DB work.

In production this becomes Redis `SET key NX EX 86400`. Same semantics, distributed. A DB unique index on `packet_hash` provides defense-in-depth.

### Problem 3: Replay attacks

An attacker captures a ciphertext and replays it later.

**Solution: Two layers.**

1. `signedAt` (epoch ms) is *inside* the AES-GCM payload — it cannot be changed without breaking the tag. The server rejects packets older than 24 hours.
2. Each payment has a unique `nonce` (UUID). Even if Alice legitimately pays Bob ₹100 twice, the nonces differ → ciphertexts differ → hashes differ → both settle. A replay of an exact copy is byte-identical and caught by the idempotency cache.

### Problem 4: Behavioral fraud (new)

SIM swap, high-velocity bursts, structuring attacks, and first-time large transfers are invisible to the cryptographic layer.

**Solution: Graph Neural Network scoring.** See the full explanation below.

---

## GNN Fraud Detection

### Why a GNN?

Traditional fraud detection treats each transaction in isolation. Real fraud is *relational* — a fraudster reuses the same device across multiple accounts, bursts many transactions in seconds, and targets new victims with large amounts. A GNN models the entire transaction graph and learns these patterns.

This implementation follows the architecture from:
> Penaganti, R. (2025). *Graph Neural Network-Based Framework for Real-Time Financial Fraud Detection in Digital Payment Ecosystems.* Journal of Computing and Data Technology.

### Graph structure

```
Nodes: User, Device, Transaction
Edges:
  user  --SENT-->        transaction
  transaction --RECEIVED_BY--> user
  user  --USES-->        device
  device --CARRIED-->    transaction
```

### Model architecture

```
Input features (5-dim per transaction):
  [0] amount_normalized       = amount / ₹10,000
  [1] hour_of_day_normalized  = hour / 24
  [2] hop_count_normalized    = hops / 8
  [3] velocity_normalized     = txns in last 10 min / 20
  [4] is_new_pair             = 1 if sender never sent to this receiver

  ↓  HeteroConv Layer 1  (SAGEConv, relation-aware message passing)
  ↓  HeteroConv Layer 2  (SAGEConv, 64-dim hidden)
  ↓  Classifier MLP      (64 → 32 → 2)
  ↓  Softmax → fraud probability [0, 1]

Loss = 0.3 × CrossEntropy(class-weighted) + 0.7 × ReconstructionMSE
```

The combined supervised + unsupervised loss (from Penaganti eq. 11) lets the model also detect *novel* fraud patterns it wasn't trained on.

### Training results

| Metric | Value |
|---|---|
| AUC-ROC | 1.00 (synthetic dataset) |
| Best Val AUC | 1.00 |
| Fraud class weight | 33× (handles 3% fraud imbalance) |
| Dataset size | 10,000 transactions, 200 users, 180 devices |

> **Note:** AUC=1.0 is expected on the synthetic dataset because fraud labels were generated from perfectly separable rules. On real UPI data expect ~0.92–0.96 (matching the Penaganti paper's reported 0.96 AUC on a 5M transaction dataset).

### Decision thresholds (configurable)

```properties
upi.gnn.block-threshold=0.85   # hard block — transaction rejected
upi.gnn.flag-threshold=0.50    # soft flag — settles, logged for review
```

### Fraud signals detected

| Signal | Feature | Threshold |
|---|---|---|
| Near-threshold structuring | `amount ≥ ₹9,800` | `f[0] > 0.98` |
| Unusual transaction hour | Before 4 AM or after 11 PM | `f[1] < 0.17 or > 0.96` |
| High hop count | More than 5 mesh hops | `f[2] > 0.625` |
| Velocity burst | More than 8 txns in 10 min | `f[3] > 0.40` |
| New sender-receiver pair | First-ever transaction | `f[4] = 1.0` |

### Python training pipeline

```
scripts/
  generate_dataset.py   → synthetic 10,000-transaction CSV
  build_graph.py        → PyTorch Geometric HeteroData graph
  model.py              → UPIFraudGNN (HeteroConv + SAGEConv)
  train.py              → training loop, early stopping, AUC eval
  export_onnx.py        → export classifier head to ONNX for Java

models/
  best_gnn.pt           → trained PyTorch weights
  fraud_classifier.onnx → ONNX export loaded by Java at runtime
  metadata.json         → threshold and AUC metadata
```

To retrain from scratch:

```bash
pip install torch torch-geometric scikit-learn pandas numpy
python scripts/generate_dataset.py
python scripts/build_graph.py
python scripts/train.py
python scripts/export_onnx.py
```

---

## File-by-file walkthrough

```
upi-offline-mesh/
├── pom.xml                          Spring Boot 3.3, ONNX Runtime 1.17, DL4J
├── models/
│   ├── best_gnn.pt                  PyTorch GNN weights
│   ├── fraud_classifier.onnx        ONNX model loaded by Java at runtime
│   └── metadata.json                AUC, threshold metadata
├── data/
│   ├── transactions.csv             10,000 synthetic labeled transactions
│   └── graph.pt                     PyTorch Geometric heterogeneous graph
├── scripts/                         Python training pipeline (see above)
└── src/main/java/com/demo/upimesh/
    │
    ├── UpiMeshApplication.java      Spring Boot entry point
    │
    ├── gnn/                         ── GNN layer (new)
    │   ├── GNNFraudScorer.java      Loads ONNX model, scores transactions
    │   └── TransactionFeatureExtractor.java  Builds 5-dim feature vector
    │
    ├── model/
    │   ├── Account.java             JPA entity (@Version optimistic lock)
    │   ├── Transaction.java         Ledger (+ gnnProbability, gnnReason fields)
    │   ├── TransactionRepository.java  + countRecentBySender, existsBySenderVpaAndReceiverVpa
    │   ├── MeshPacket.java          Wire format (outer fields readable, ciphertext opaque)
    │   └── PaymentInstruction.java  Decrypted payload
    │
    ├── crypto/
    │   ├── ServerKeyHolder.java     RSA-2048 keypair on startup
    │   └── HybridCryptoService.java RSA-OAEP + AES-256-GCM + ciphertext hash
    │
    ├── service/
    │   ├── BridgeIngestionService.java  Pipeline: hash→idempotency→decrypt→freshness→GNN→settle
    │   ├── SettlementService.java       @Transactional debit+credit (+ stores GNN metadata)
    │   ├── IdempotencyService.java      ConcurrentHashMap SETNX (Redis-equivalent)
    │   ├── MeshSimulatorService.java    Gossip protocol simulation
    │   ├── DemoService.java             Seeds accounts, simulates sender phone
    │   └── VirtualDevice.java           One simulated phone in the mesh
    │
    └── controller/
        ├── ApiController.java       REST endpoints (flush now returns fraudProbability)
        └── DashboardController.java Serves dashboard at /
```

---

## API reference

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Dashboard HTML (with GNN score columns) |
| GET | `/api/server-key` | Server's RSA public key (base64) |
| GET | `/api/accounts` | All accounts and balances |
| GET | `/api/transactions` | Last 20 transactions (includes `gnnProbability`, `gnnReason`) |
| GET | `/api/mesh/state` | State of every virtual device |
| POST | `/api/demo/send` | Simulate sender phone — encrypt + inject |
| POST | `/api/mesh/gossip` | One gossip round |
| POST | `/api/mesh/flush` | Bridges upload (parallel) — response includes `fraudProbability` |
| POST | `/api/mesh/reset` | Clear mesh + idempotency cache |
| POST | `/api/bridge/ingest` | **Production endpoint** — real bridges POST here |
| GET | `/h2-console` | Browse the in-memory H2 database |

### `/api/bridge/ingest` response (updated)

```json
{
  "outcome": "SETTLED",
  "packetHash": "a3f8c9...",
  "reason": "nominal",
  "transactionId": 3,
  "fraudProbability": 0.023,
  "gnnReason": "nominal"
}
```

Possible `outcome` values: `SETTLED`, `DUPLICATE_DROPPED`, `INVALID`, `BLOCKED_BY_GNN`, `REJECTED`.

---

## Tests

```cmd
mvnw.cmd test
```

Three tests, all testing security-critical paths:

| Test | What it proves |
|---|---|
| `encryptDecryptRoundTrip` | Hybrid encryption is symmetric and lossless |
| `tamperedCiphertextIsRejected` | A single flipped bit returns `INVALID`, not a silent wrong settlement |
| `singlePacketDeliveredByThreeBridgesSettlesExactlyOnce` | Three concurrent bridge deliveries → exactly 1 `SETTLED`, 2 `DUPLICATE_DROPPED`, sender debited once |

The concurrency test is the headline — it fires three threads at `BridgeIngestionService.ingest()` simultaneously with a `CountDownLatch` for maximum overlap.

---

## What's NOT real (and what would change for production)

| What's in the demo | What it would be in production |
|---|---|
| H2 in-memory DB | PostgreSQL / MySQL with replicas |
| `ConcurrentHashMap` idempotency | Redis `SET NX EX` |
| RSA keypair regenerated on startup | Private key in HSM (AWS KMS, HashiCorp Vault) |
| Server-side `DemoService.createPacket()` | Same code on Android in Kotlin |
| Software mesh simulation | Real BLE GATT or Wi-Fi Direct |
| ONNX classifier head only | Full GNN forward pass in Python sidecar or TorchServe |
| Synthetic training data | Real NPCI transaction logs with verified fraud labels |
| Hand-crafted embedding projection | Full `R-GCN + GAT + TGN` graph forward pass (Penaganti architecture) |
| No auth on `/api/bridge/ingest` | Mutual TLS + signed bridge-node certificates |
| GNN block threshold hardcoded | Dynamic threshold from A/B testing on false-positive rate |

The cryptography, idempotency, and GNN pipeline structure are production-shaped. The infrastructure around them is what changes.

---

## Honest limitations of the concept

1. **The receiver cannot verify funds exist offline.** The payment is an IOU until the packet reaches the backend. If the sender's balance is empty by then, the settlement is `REJECTED`. Real offline UPI (UPI Lite) uses a pre-funded hardware-backed wallet to give cryptographic proof of available funds.

2. **A malicious sender can double-spend offline.** ₹500 in their account → they send it to Bob in basement A, walk to basement B, send it to Carol. First packet to reach the backend wins; the other is `REJECTED`.

3. **GNN block threshold is conservative by design.** `0.85` means the model is very confident before blocking. Lowering it catches more fraud but increases false positives. The right value depends on the business's tolerance for blocking legitimate payments.

4. **Bluetooth in real life is hard.** Background BLE on Android is throttled since Android 8. iOS peripheral mode is locked down. This demo sidesteps these entirely by simulating the mesh.

5. **Privacy / liability.** A stranger carries your encrypted transaction packet. They can't read it, but its existence is metadata. Real deployment requires regulatory disclosure and handling of seized devices.

---

## Contributors

<a href="https://github.com/SparshKapoor-CODER/GNN_UPI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SparshKapoor-CODER/GNN_UPI" />
</a>

---

## License

Demo and research code. No license. Use it for learning, projects, and presentations.

---

## Papers referenced

- Penaganti, R. (2025). *Graph Neural Network-Based Framework for Real-Time Financial Fraud Detection in Digital Payment Ecosystems.* Journal of Computing and Data Technology, 1(2), 91–97. https://doi.org/10.71426/jcdt.v1.i2.pp91-97
- Mungara, D., Ramulu, H. S., & Acar, Y. (2025). *Security and Privacy Advice for UPI Users in India.* 34th USENIX Security Symposium.
- Bhavani, C. N. (2024). *Analysis on Fraudulent Threats and Mitigating Strategies in UPI Transactions.* 12th International Conference on Emerging Trends in Corporate Finance and Financial Markets.