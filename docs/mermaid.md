---
config:
  look: neo
  theme: neutral
---
flowchart TD
    Input["Input Contact Matrix\n(B×1×256×256)"]
    AssayID["Assay ID\n[B] (4 assays)"]
    AssayEmbed["Assay Embedding\n(4 assays → 8)"]
    FiLM["FiLMLayer\nγ, β projections\n8→32"]
    Stage1["Stage1 Conv2d\n1→32, k=7, s=2\n(B×32×128×128)"]
    Stage2["Stage2 ResBlock\n32→64, s=2\n(B×64×64×64)"]
    Stage3["Stage3 ResBlock\n64→128, s=2\n(B×128×32×32)"]
    Stage4["Stage4 ResBlock\n128→256, s=1\n(B×256×32×32)"]
    Reshape["Reshape\n(B×256×32×32) → (B×1024×256)"]
    Masker["VanillaMasker\nScorer MLP\n256→64→1"]
    TempControl["Temperature τ_f\n1.0→0.5 over epochs"]
    Gumbel["Gumbel-Softmax\nTraining: stochastic\nInference: deterministic"]
    VisibleTokens["Visible Tokens\n(B×512×256)"]
    VisibleIndices["Visible Indices\n[B×512]"]
    Codebook["EMA Codebook\n512×256"]
    VQ["EMAVectorQuantizer\nNearest neighbor"]
    Quantized["Quantized Tokens\n(B×512×256)"]
    CommitLoss["Commitment Loss\nMSE(z_e, z_q)"]
    CodeIndices["Code Indices\n[B×512]"]
    FingerprintPool["Pooling Strategy\nMean or Attention"]
    FPProjection["FP Projection\n256→32"]
    Fingerprint["Fingerprint\n[B×32]"]
    Histogram["Histogram\n[B×512] (optional)"]
    MaskTokens["Mask Tokens\nLearnable\n[1×1×256]"]
    PosEmbed["Positional Embedding\n[1×1024×256]"]
    Scatter["Scatter Tokens\nBuild full sequence"]
    InputProj["Input Projection\n256→128"]
    TransformerBlocks["4× TransformerBlocks\nMHSA + FFN\ndim=128"]
    OutputProj["Output Projection\n128→256"]
    DemaskedOut["Demasked Features\n(B×1024×256)"]
    SpatialReshape["Spatial Reshape\n(B×1024×256) → (B×256×32×32)"]
    Up1["UpsampleBlock 1\nResBlock 256→128\n2× bilinear upsample\n(B×128×64×64)"]
    Up2["UpsampleBlock 2\nResBlock 128→64\n2× bilinear upsample\n(B×64×128×128)"]
    Up3["UpsampleBlock 3\nResBlock 64→32\n2× bilinear upsample\n(B×32×256×256)"]
    ContactHead["ContactReconHead\nConv 32→16→1\n(B×1×256×256)"]
    CellClassifier["CellClassifierHead\nMLP 256→128→64→16\n[B×16]"]
    Reconstructed["Reconstructed Contact\n(B×1×256×256)"]
    CellLogits["Cell Type Logits\n[B×16]"]
    NoMasking["Ablation: No Masking\nAll 1024 tokens"]
    NoFiLM["Ablation: No FiLM\nZero assay embedding"]
    Input --> Stage1
    AssayID --> AssayEmbed
    AssayEmbed --> FiLM
    FiLM -->|"modulate features"| Stage1
    Stage1 --> Stage2 --> Stage3 --> Stage4 --> Reshape
    Reshape --> Masker
    Reshape --> NoMasking
    NoMasking --> VQ
    Masker --> TempControl
    TempControl --> Gumbel
    Gumbel --> VisibleTokens
    Gumbel --> VisibleIndices
    VisibleTokens --> VQ
    Codebook --> VQ
    VQ --> Quantized
    VQ --> CommitLoss
    VQ --> CodeIndices
    Quantized --> FingerprintPool
    FingerprintPool --> FPProjection
    FPProjection --> Fingerprint
    CodeIndices --> Histogram
    Quantized --> Scatter
    VisibleIndices --> Scatter
    MaskTokens --> Scatter
    PosEmbed --> Scatter
    Scatter --> InputProj
    InputProj --> TransformerBlocks
    TransformerBlocks --> OutputProj
    OutputProj --> DemaskedOut
    DemaskedOut --> SpatialReshape
    SpatialReshape --> Up1 --> Up2 --> Up3
    Up3 --> ContactHead
    ContactHead --> Reconstructed
    VisibleTokens -->|"mean pool"| CellClassifier
    CellClassifier --> CellLogits
    AssayEmbed --> NoFiLM
    NoFiLM -->|"zero modulation"| Stage1
    style Input fill:#1a1a2e,color:#eee
    style AssayID fill:#1a1a2e,color:#eee
    style AssayEmbed fill:#16213e,color:#eee
    style FiLM fill:#16213e,color:#eee
    style Stage1 fill:#0f3460,color:#eee
    style Stage2 fill:#0f3460,color:#eee
    style Stage3 fill:#0f3460,color:#eee
    style Stage4 fill:#0f3460,color:#eee
    style Reshape fill:#0f3460,color:#eee
    style Masker fill:#2b4162,color:#eee
    style TempControl fill:#2b4162,color:#eee
    style Gumbel fill:#2b4162,color:#eee
    style VisibleTokens fill:#2b4162,color:#eee
    style VisibleIndices fill:#2b4162,color:#eee
    style Codebook fill:#e94560,color:#fff
    style VQ fill:#e94560,color:#fff
    style Quantized fill:#e94560,color:#fff
    style CommitLoss fill:#e94560,color:#fff
    style CodeIndices fill:#e94560,color:#fff
    style FingerprintPool fill:#f39c12,color:#fff
    style FPProjection fill:#f39c12,color:#fff
    style Fingerprint fill:#f39c12,color:#fff
    style Histogram fill:#f39c12,color:#fff
    style MaskTokens fill:#0a3d62,color:#eee
    style PosEmbed fill:#0a3d62,color:#eee
    style Scatter fill:#0a3d62,color:#eee
    style InputProj fill:#0a3d62,color:#eee
    style TransformerBlocks fill:#0a3d62,color:#eee
    style OutputProj fill:#0a3d62,color:#eee
    style DemaskedOut fill:#0a3d62,color:#eee
    style SpatialReshape fill:#1e3799,color:#eee
    style Up1 fill:#1e3799,color:#eee
    style Up2 fill:#1e3799,color:#eee
    style Up3 fill:#1e3799,color:#eee
    style ContactHead fill:#079992,color:#fff
    style CellClassifier fill:#079992,color:#fff
    style Reconstructed fill:#b8e994,color:#000
    style CellLogits fill:#b8e994,color:#000
    style NoMasking fill:#7f8c8d,color:#fff,stroke-dasharray: 5 5
    style NoFiLM fill:#7f8c8d,color:#fff,stroke-dasharray: 5 5