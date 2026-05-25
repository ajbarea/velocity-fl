---
title: Documentation
hide:
  - navigation
  - toc
  - footer
---

<div class="hero" markdown>

# VelocityFL

**The uv of Federated Learning — Rust speed, Python ergonomics.**
{ .hero-subtitle }

<div class="hero-buttons" markdown>

[:octicons-rocket-24: Get Started](getting-started.md){ .md-button .md-button--primary }
[:octicons-book-24: Architecture](architecture.md){ .md-button }

</div>

<div class="hero-tagline" markdown>

:octicons-cpu-24: Rust Core | :octicons-code-24: Python API | :octicons-graph-24: [4–10× Faster Aggregation](benchmarks.md)
{ .hero-modes }

</div>

</div>

<div class="scroll-hint" aria-hidden="true">
  <div class="scroll-chevron"></div>
</div>

<section class="landing-section landing-section--intro">
  <div class="section-inner">
    <h2 class="section-title">What Is VelocityFL?</h2>
    <p class="section-lead">VelocityFL is a federated learning orchestration framework with a <strong>Rust core</strong> for hot-path aggregation and attack simulation, wrapped in a <strong>Python-first API</strong> for researchers using Hugging Face, PEFT, and PyTorch. Configure a server, run rounds, inspect metrics &mdash; every hot loop runs in compiled code.</p>
  </div>
</section>

<section class="landing-section landing-section--promise">
  <div class="section-inner">
    <h2 class="section-title">Rust core, Python surface</h2>
    <p class="section-lead">A PyO3 extension runs the aggregation math; a Typer CLI and a thin Python API keep experimentation friction low. Prefect wraps each round so you get flow-level visibility without custom logging glue. The aggregation kernel runs <a href="benchmarks/">4–10× faster</a> than the pure-Python fallback, widening with parameter count (6.9× at 1M params, 10.4× at 10M, <code>FedAvg</code>) &mdash; the measured claim is on aggregation only, not end-to-end training, with honest caveats on PyO3 marshaling cost.</p>
  </div>
</section>

<section class="landing-section">
  <div class="section-inner">
    <h2 class="section-title">One API, Full Pipeline</h2>
    <div class="pipeline-flow">
      <div class="pipeline-step">
        <span class="step-icon material-symbols-outlined">settings</span>
        <span class="step-label">Config</span>
      </div>
      <div class="pipeline-step">
        <span class="step-icon material-symbols-outlined">group</span>
        <span class="step-label">Clients</span>
      </div>
      <div class="pipeline-step">
        <span class="step-icon material-symbols-outlined">memory</span>
        <span class="step-label">Rust Core</span>
      </div>
      <div class="pipeline-step">
        <span class="step-icon material-symbols-outlined">scatter_plot</span>
        <span class="step-label">Aggregate</span>
      </div>
      <div class="pipeline-step">
        <span class="step-icon material-symbols-outlined">bug_report</span>
        <span class="step-label">Attack Sim</span>
      </div>
      <div class="pipeline-step">
        <span class="step-icon material-symbols-outlined">trending_up</span>
        <span class="step-label">Metrics</span>
      </div>
    </div>
    <p class="pipeline-caption">Python config &rarr; federated clients &rarr; Rust aggregation &rarr; Prefect-tracked round summaries</p>
  </div>
</section>

<section class="landing-section">
  <div class="section-inner">
    <h2 class="section-title">Explore the Docs</h2>
    <div class="feature-grid">
      <a href="getting-started/" class="feature-card" style="--card-accent: #7c3aed">
        <span class="feature-icon material-symbols-outlined">rocket_launch</span>
        <div class="feature-name">Getting Started</div>
        <p>Install with <code>maturin develop</code>, run your first round in minutes.</p>
      </a>
      <a href="cli/" class="feature-card" style="--card-accent: #8b5cf6">
        <span class="feature-icon material-symbols-outlined">terminal</span>
        <div class="feature-name">CLI Reference</div>
        <p>Every <code>velocity</code> command &mdash; <code>run</code>, <code>simulate-attack</code>, <code>strategies</code>.</p>
      </a>
      <a href="architecture/" class="feature-card" style="--card-accent: #a855f7">
        <span class="feature-icon material-symbols-outlined">account_tree</span>
        <div class="feature-name">Architecture</div>
        <p>How the Rust crate, PyO3 bindings, and Python orchestrator fit together.</p>
      </a>
      <a href="configuration/" class="feature-card" style="--card-accent: #c084fc">
        <span class="feature-icon material-symbols-outlined">tune</span>
        <div class="feature-name">Configuration</div>
        <p>Server options, strategy knobs, and attack parameters &mdash; every field explained.</p>
      </a>
      <a href="strategies/" class="feature-card" style="--card-accent: #d8b4fe">
        <span class="feature-icon material-symbols-outlined">hub</span>
        <div class="feature-name">Strategies</div>
        <p>FedAvg, FedProx, FedMedian, TrimmedMean, Krum, Multi-Krum, Bulyan, GeometricMedian &mdash; and where to add your own.</p>
      </a>
      <a href="attacks/" class="feature-card" style="--card-accent: #e9d5ff">
        <span class="feature-icon material-symbols-outlined">bug_report</span>
        <div class="feature-name">Attacks</div>
        <p>Model poisoning, Sybil nodes, Gaussian noise, label flipping.</p>
      </a>
      <a href="api/" class="feature-card" style="--card-accent: #a855f7">
        <span class="feature-icon material-symbols-outlined">api</span>
        <div class="feature-name">API Reference</div>
        <p>Python surface: <code>VelocityServer</code>, <code>Strategy</code>, <code>ClientUpdate</code>.</p>
      </a>
    </div>
  </div>
</section>

<section class="landing-section">
  <div class="section-inner">
    <h2 class="section-title">Key Features</h2>
    <div class="feature-grid feature-grid--compact">
      <div class="feature-card feature-card--flat">
        <div class="feature-name"><span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 6px;">bolt</span> Rust Hot Path</div>
        <p>Aggregation, median, Krum/Multi-Krum, and attack simulation compiled to Rust via PyO3. <a href="benchmarks/">4–10× faster</a> aggregation than the pure-Python fallback, widening with parameter count (<code>FedAvg</code>: 77.8 ms vs 538 ms at 1M params, 817 ms vs 8.48 s at 10M); PyO3 list-marshaling on the return path is the remaining FFI overhead (tracked).</p>
      </div>
      <div class="feature-card feature-card--flat">
        <div class="feature-name"><span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 6px;">science</span> Research-Friendly</div>
        <p>Python API designed around Hugging Face model IDs, PEFT adapters, and PyTorch tensors. Pure-Python fallback when the Rust extension is unavailable.</p>
      </div>
      <div class="feature-card feature-card--flat">
        <div class="feature-name"><span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 6px;">visibility</span> Observable by Default</div>
        <p>Prefect-native flows expose round metrics, client updates, and attack results as first-class events &mdash; no custom logging glue needed.</p>
      </div>
      <div class="feature-card feature-card--flat">
        <div class="feature-name"><span class="material-symbols-outlined" style="vertical-align: middle; margin-right: 6px;">shield</span> Resilience Testing</div>
        <p>Register Byzantine attacks alongside your strategy and watch how aggregation holds up, round by round.</p>
      </div>
    </div>
  </div>
</section>

<section class="landing-section">
  <div class="section-inner">
    <h2 class="section-title">Technology Stack</h2>
    <div class="tech-stack">
      <div class="tech-item"><strong>Core</strong> Rust + PyO3</div>
      <div class="tech-item"><strong>API</strong> Python</div>
      <div class="tech-item"><strong>Build</strong> maturin + uv</div>
      <div class="tech-item"><strong>Orchestration</strong> Prefect</div>
      <div class="tech-item"><strong>CLI</strong> Typer</div>
      <div class="tech-item"><strong>Validation</strong> Pydantic</div>
      <div class="tech-item"><strong>ML</strong> HuggingFace + PEFT + Torch</div>
      <div class="tech-item"><strong>Docs</strong> Zensical</div>
    </div>
  </div>
</section>

<section class="landing-section cta-section">
  <div class="section-inner">
    <h2 class="section-title">Get started</h2>
    <p class="cta-lead">Clone, <code>maturin develop</code>, run your first round.</p>
    <a href="getting-started/" class="md-button md-button--primary cta-button">Read the Quickstart</a>
  </div>
</section>

<footer class="landing-footer">
  <span>AJ Barea · 2026</span>
  <a href="https://github.com/ajbarea/velocity-fl" aria-label="GitHub">
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
  </a>
</footer>
