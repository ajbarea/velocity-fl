---
title: Documentation
hide:
  - navigation
  - toc
  - footer
---

<div class="hero" markdown>

# Velocity-FL

**The uv of Federated Learning: Rust speed, Python ergonomics.**
{ .hero-subtitle }

<div class="hero-buttons" markdown>

[:octicons-rocket-24: Get Started](getting-started.md){ .md-button .md-button--primary }
[:octicons-book-24: Architecture](architecture.md){ .md-button }

</div>

<div class="hero-chips" markdown>
  <span class="chip" markdown="span">:octicons-cpu-24: Rust core</span>
  <span class="chip" markdown="span">:octicons-code-24: Python API</span>
  <a class="chip chip--link" href="benchmarks/" markdown="span">:octicons-graph-24: Up to 138&times; faster FedAvg</a>
</div>

</div>

<div class="scroll-hint" aria-hidden="true">
  <div class="scroll-chevron"></div>
</div>

<section class="landing-section landing-section--intro">
  <div class="section-inner">
    <h2 class="section-title">What is Velocity-FL?</h2>
    <p class="section-lead">Federated learning where the round's hot path, aggregation and attack simulation, runs in <strong>Rust</strong>, and everything you touch stays <strong>Python</strong>: Hugging Face, PEFT and PyTorch.</p>
  </div>
</section>

<section class="landing-section landing-section--promise">
  <div class="section-inner">
    <h2 class="section-title">Aggregation in compiled code</h2>
    <div class="stat-row">
      <div class="stat">
        <div class="stat-value">4.0 ms <span class="stat-vs">vs 545 ms</span></div>
        <div class="stat-label"><code>FedAvg</code> at 1M params</div>
      </div>
      <div class="stat">
        <div class="stat-value">42.2 ms <span class="stat-vs">vs 5.82 s</span></div>
        <div class="stat-label"><code>FedAvg</code> at 10M params</div>
      </div>
      <div class="stat">
        <div class="stat-value">~138&times;</div>
        <div class="stat-label">faster than the pure-Python fallback</div>
      </div>
    </div>
    <p class="stat-note">Aggregation only, not end-to-end training, on the latest idle-box snapshot. <a href="benchmarks/">Methodology and every tier</a></p>
  </div>
</section>

<section class="landing-section">
  <div class="section-inner">
    <h2 class="section-title">One round, end to end</h2>
    <ol class="round-flow">
      <li class="round-step">
        <span class="step-icon material-symbols-outlined">tune</span>
        <span class="step-label">Configure</span>
        <span class="step-text">A server, a strategy and a round count, in Python or TOML.</span>
      </li>
      <li class="round-step">
        <span class="step-icon material-symbols-outlined">group</span>
        <span class="step-label">Collect</span>
        <span class="step-text">Client updates cross into Rust, zero-copy where possible.</span>
      </li>
      <li class="round-step">
        <span class="step-icon material-symbols-outlined">memory</span>
        <span class="step-label">Aggregate</span>
        <span class="step-text">Nine strategies, all in Rust, from FedAvg to Bulyan.</span>
      </li>
      <li class="round-step">
        <span class="step-icon material-symbols-outlined">shield</span>
        <span class="step-label">Attack and record</span>
        <span class="step-text">Byzantine attacks run in the core; Prefect records every round.</span>
      </li>
    </ol>
  </div>
</section>

<section class="landing-section">
  <div class="section-inner">
    <h2 class="section-title">Explore the docs</h2>
    <div class="feature-grid">
      <a href="getting-started/" class="feature-card" style="--card-accent: #7c3aed">
        <span class="feature-icon material-symbols-outlined">rocket_launch</span>
        <div class="feature-name">Getting Started</div>
        <p>Install and run your first round.</p>
      </a>
      <a href="cli/" class="feature-card" style="--card-accent: #8b5cf6">
        <span class="feature-icon material-symbols-outlined">terminal</span>
        <div class="feature-name">CLI Reference</div>
        <p>Every <code>velocity</code> command.</p>
      </a>
      <a href="architecture/" class="feature-card" style="--card-accent: #9333ea">
        <span class="feature-icon material-symbols-outlined">account_tree</span>
        <div class="feature-name">Architecture</div>
        <p>The Rust crate, PyO3 bindings and Python layer.</p>
      </a>
      <a href="configuration/" class="feature-card" style="--card-accent: #a855f7">
        <span class="feature-icon material-symbols-outlined">settings</span>
        <div class="feature-name">Configuration</div>
        <p>Every server, strategy and attack field.</p>
      </a>
      <a href="strategies/" class="feature-card" style="--card-accent: #c026d3">
        <span class="feature-icon material-symbols-outlined">hub</span>
        <div class="feature-name">Strategies</div>
        <p>The nine aggregators, and which to pick.</p>
      </a>
      <a href="attacks/" class="feature-card" style="--card-accent: #db2777">
        <span class="feature-icon material-symbols-outlined">bug_report</span>
        <div class="feature-name">Attacks</div>
        <p>Poisoning, Sybils, noise, label flipping.</p>
      </a>
      <a href="api/" class="feature-card" style="--card-accent: #6366f1">
        <span class="feature-icon material-symbols-outlined">api</span>
        <div class="feature-name">API Reference</div>
        <p><code>VelocityServer</code>, <code>Strategy</code>, <code>ClientUpdate</code>.</p>
      </a>
      <a href="benchmarks/" class="feature-card" style="--card-accent: #4f46e5">
        <span class="feature-icon material-symbols-outlined">speed</span>
        <div class="feature-name">Benchmarks</div>
        <p>Rust against Python, tier by tier.</p>
      </a>
    </div>
    <div class="stack-chips" aria-label="Built with">
      <span>Rust + PyO3</span><span>maturin + uv</span><span>Prefect</span><span>Typer</span><span>Pydantic</span><span>Hugging Face</span><span>PEFT</span><span>PyTorch</span>
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
  <span>2026 <img src="assets/brand.png" alt="" aria-hidden="true" class="brand-mark"> AJ Barea</span>
  <a href="https://github.com/ajbarea/velocity-fl" aria-label="GitHub">
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
  </a>
</footer>
