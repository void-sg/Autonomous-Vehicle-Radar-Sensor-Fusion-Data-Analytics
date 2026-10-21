import json
import os
import subprocess
import time

def extract_notebook_images(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    images = {}
    for i, cell in enumerate(nb['cells']):
        for j, out in enumerate(cell.get('outputs', [])):
            data = out.get('data', {})
            if 'image/png' in data:
                key = f"cell_{i}_out_{j}"
                images[key] = data['image/png']
    return images

def build_handbook_html(images):
    # Mapping images
    img_target_dist   = images.get('cell_7_out_0', '')
    img_univariate    = images.get('cell_8_out_0', '')
    img_bivariate     = images.get('cell_9_out_0', '')
    img_corr_heatmap  = images.get('cell_10_out_0', '')
    img_dist_fits     = images.get('cell_12_out_1', '')
    img_clt           = images.get('cell_14_out_0', '')
    img_hypo_tests    = images.get('cell_16_out_1', '')
    img_residuals     = images.get('cell_19_out_2', '')
    img_anomalies     = images.get('cell_22_out_2', '')
    img_time_series   = images.get('cell_24_out_0', '')
    img_acf           = images.get('cell_24_out_1', '')
    img_ml_eval       = images.get('cell_26_out_1', '')

    # Using raw template to avoid f-string curly brace conflicts
    template = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Autonomous Vehicle Radar, Sensor Fusion & Data Analytics Handbook</title>
<style>
    @page {
        size: A4;
        margin: 18mm 16mm 20mm 16mm;
        @bottom-right {
            content: counter(page);
            font-size: 9pt;
            font-family: 'Segoe UI', system-ui, sans-serif;
            color: #64748b;
        }
    }
    
    body {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
        color: #1e293b;
        line-height: 1.6;
        font-size: 10.5pt;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }

    .cover-page {
        page-break-after: always;
        height: 100vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 40px 20px;
        box-sizing: border-box;
        border: 4px solid #1e3a8a;
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
    }

    .cover-title {
        font-size: 26pt;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 12px;
        line-height: 1.25;
    }

    .cover-subtitle {
        font-size: 14pt;
        font-weight: 500;
        color: #0369a1;
        margin-bottom: 30px;
        max-width: 650px;
        line-height: 1.4;
    }

    .cover-badge {
        display: inline-block;
        background-color: #1e3a8a;
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 10.5pt;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 40px;
    }

    .cover-meta {
        margin-top: 50px;
        border-top: 2px solid #cbd5e1;
        padding-top: 25px;
        font-size: 10.5pt;
        color: #475569;
        line-height: 1.8;
    }

    .page-break {
        page-break-after: always;
    }

    h1 {
        color: #1e3a8a;
        font-size: 18pt;
        font-weight: 700;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 6px;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    h2 {
        color: #0369a1;
        font-size: 13.5pt;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    h3 {
        color: #0f766e;
        font-size: 11.5pt;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 6px;
    }

    p {
        margin-top: 0;
        margin-bottom: 10px;
        text-align: justify;
    }

    ul, ol {
        margin-top: 0;
        margin-bottom: 12px;
        padding-left: 22px;
    }

    li {
        margin-bottom: 5px;
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 9pt;
    }

    th, td {
        border: 1px solid #cbd5e1;
        padding: 7px 10px;
        text-align: left;
    }

    th {
        background-color: #f1f5f9;
        color: #1e293b;
        font-weight: 600;
    }

    tr:nth-child(even) {
        background-color: #f8fafc;
    }

    .callout {
        border-left: 4px solid #3b82f6;
        background-color: #eff6ff;
        padding: 12px 16px;
        margin: 14px 0;
        border-radius: 0 6px 6px 0;
        font-size: 10pt;
    }

    .callout-title {
        font-weight: 700;
        color: #1e40af;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .callout-analogy {
        border-left-color: #10b981;
        background-color: #ecfdf5;
    }
    .callout-analogy .callout-title {
        color: #047857;
    }

    .callout-formula {
        border-left-color: #8b5cf6;
        background-color: #f5f3ff;
    }
    .callout-formula .callout-title {
        color: #6d28d9;
    }

    .callout-warning {
        border-left-color: #ef4444;
        background-color: #fef2f2;
    }
    .callout-warning .callout-title {
        color: #b91c1c;
    }

    .figure-container {
        text-align: center;
        margin: 16px 0;
        page-break-inside: avoid;
    }

    .figure-container img {
        max-width: 95%;
        height: auto;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }

    .figure-caption {
        font-size: 9pt;
        color: #64748b;
        font-style: italic;
        margin-top: 6px;
    }

    code {
        font-family: Consolas, 'Courier New', Courier, monospace;
        background-color: #f1f5f9;
        padding: 1px 4px;
        border-radius: 3px;
        font-size: 9.5pt;
        color: #0f172a;
    }

    pre {
        background-color: #0f172a;
        color: #f8fafc;
        padding: 12px;
        border-radius: 6px;
        font-size: 8.5pt;
        font-family: Consolas, 'Courier New', Courier, monospace;
        overflow-x: auto;
        margin: 12px 0;
        line-height: 1.45;
    }

    .toc {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 18px 24px;
        border-radius: 8px;
        margin: 20px 0;
    }

    .toc ul {
        list-style-type: none;
        padding-left: 0;
    }

    .toc li {
        margin-bottom: 8px;
        font-size: 10pt;
    }

    .toc a {
        color: #2563eb;
        text-decoration: none;
        font-weight: 500;
    }

    .math-box {
        background-color: #f8fafc;
        border: 1px dashed #94a3b8;
        padding: 10px 14px;
        border-radius: 6px;
        font-family: 'Cambria Math', 'Times New Roman', serif;
        font-size: 11pt;
        text-align: center;
        margin: 10px 0;
    }
</style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover-page">
    <div class="cover-badge">STATISTICAL DATA ANALYTICS (SDA) HANDBOOK</div>
    <div class="cover-title">Autonomous Vehicle Radar, Sensor Fusion & Data Analytics</div>
    <div class="cover-subtitle">From Physical Sensing and Radio Frequency Waves to Rigorous Statistical Inference and Machine Learning Anomaly Detection</div>
    
    <div style="font-size: 12pt; color: #334155; margin-top: 15px; font-weight: 600;">
        A Complete Zero-to-Hero Foundational & Applied Guide for Engineers
    </div>

    <div class="cover-meta">
        <strong>Academic Context:</strong> SY B.Tech Electronics & Computer Engineering<br>
        <strong>Subject:</strong> Statistical Data Analytics (SDA) | AY 2026–27 Semester I<br>
        <strong>Dataset Examined:</strong> <code>autonomous_radar_telemetry_50k.csv</code> (50,000 Frames × 20 Telemetry Features)<br>
        <strong>Target Investigation:</strong> Predicting Sensor Degradation & Early Fault Detection in Autonomous Vehicles
    </div>
</div>

<!-- TABLE OF CONTENTS -->
<div class="page-break">
    <h1>Table of Contents</h1>
    <div class="toc">
        <ul>
            <li><strong>1. The Perception Problem: How Autonomous Vehicles See the World</strong>
                <ul>
                    <li>1.1 Levels of Autonomous Driving</li>
                    <li>1.2 The Perception Triad: Camera vs. LiDAR vs. Radar</li>
                    <li>1.3 What is Sensor Fusion and Why is it Necessary?</li>
                </ul>
            </li>
            <li><strong>2. Radar Physics & Embedded Hardware Architecture</strong>
                <ul>
                    <li>2.1 How Millimeter-Wave (mmWave) FMCW Radar Operates</li>
                    <li>2.2 The Physics of Range and Doppler Radial Velocity</li>
                    <li>2.3 The Embedded Electronics Chain: LNA, ADC, DSP, MCU</li>
                    <li>2.4 The Vehicle's Nervous System: CAN Bus Communication & Latency</li>
                    <li>2.5 What is Sensor Degradation and Why Does It Occur?</li>
                </ul>
            </li>
            <li><strong>3. Deconstructing the 20 Telemetry Features</strong>
                <ul>
                    <li>3.1 Radar Physics / RF Sensing (10 Features)</li>
                    <li>3.2 Hardware & System Health (5 Features)</li>
                    <li>3.3 Environmental Context (2 Features)</li>
                    <li>3.4 Vehicle Context & Other Sensors (2 Features)</li>
                    <li>3.5 The Target Variable: Sensor Degradation Flag</li>
                </ul>
            </li>
            <li><strong>4. Data Science & Descriptive Statistics from Scratch</strong>
                <ul>
                    <li>4.1 What is Tabular Telemetry Data?</li>
                    <li>4.2 Measures of Central Tendency: Mean vs. Median vs. Mode</li>
                    <li>4.3 Measures of Dispersion: Variance, Standard Deviation, and IQR</li>
                    <li>4.4 Shape of Data: Skewness and Kurtosis</li>
                    <li>4.5 The Five-Number Summary and How to Read Boxplots</li>
                </ul>
            </li>
            <li><strong>5. Continuous Probability Distributions & Theoretical Modeling</strong>
                <ul>
                    <li>5.1 What is a Probability Density Function (PDF)?</li>
                    <li>5.2 The Gaussian / Normal Distribution: Electronic Noise & Frequencies</li>
                    <li>5.3 The Exponential Distribution: Atmospheric Rainfall Arrival</li>
                    <li>5.4 The Continuous Uniform Distribution: Vehicle Speeds</li>
                    <li>5.5 Quantile-Quantile (Q-Q) Plots and Kolmogorov-Smirnov (K-S) Tests</li>
                </ul>
            </li>
            <li><strong>6. Statistical Estimation & The Central Limit Theorem (CLT)</strong>
                <ul>
                    <li>6.1 Populations vs. Samples</li>
                    <li>6.2 The Central Limit Theorem Explained Simply</li>
                    <li>6.3 Standard Error of the Mean (SE)</li>
                    <li>6.4 Constructing 95% and 99% Confidence Intervals (CI)</li>
                </ul>
            </li>
            <li><strong>7. Statistical Hypothesis Testing Demystified</strong>
                <ul>
                    <li>7.1 The 6-Step Scientific Decision-Making Framework</li>
                    <li>7.2 Null vs. Alternative Hypotheses and p-Values</li>
                    <li>7.3 One-Sample t-Test: Radar Frequency Standards Compliance</li>
                    <li>7.4 Two-Sample Welch's t-Test: Thermal Profiling of Degraded Sensors</li>
                    <li>7.5 One-Way ANOVA: Latency Independence across Speed Regimes</li>
                    <li>7.6 Chi-Square Test of Independence & Odds Ratios</li>
                </ul>
            </li>
            <li><strong>8. Correlation, Regression & Radar Path Loss</strong>
                <ul>
                    <li>8.1 Correlation vs. Causation: Pearson r and Spearman rho</li>
                    <li>8.2 Simple Linear Regression: Quantifying Radar Path Loss (-0.10 dB/m)</li>
                    <li>8.3 Multiple Linear Regression & R² Variance Explained</li>
                    <li>8.4 Residual Diagnostics: Normality and Homoscedasticity</li>
                </ul>
            </li>
            <li><strong>9. Electronic Anomaly Detection: Z-Score vs. IQR</strong>
                <ul>
                    <li>9.1 Parametric Z-Score Detection (|Z| > 3.0)</li>
                    <li>9.2 Non-Parametric Interquartile Range (1.5 x IQR) Rule</li>
                    <li>9.3 Profiling Electronic Power Rail & Temperature Anomalies</li>
                </ul>
            </li>
            <li><strong>10. Sequential Telemetry & Time-Series Analysis</strong>
                <ul>
                    <li>10.1 Frame-by-Frame Temporal Dynamics & Rolling Moving Averages</li>
                    <li>10.2 Volatility Bands (+/- 2-sigma) and Autocorrelation Function (ACF)</li>
                </ul>
            </li>
            <li><strong>11. Predictive Machine Learning for Early Sensor Degradation</strong>
                <ul>
                    <li>11.1 Supervised Binary Classification using Logistic Regression</li>
                    <li>11.2 Train/Test Split, Standardization, and the Sigmoid Function</li>
                    <li>11.3 Evaluation Metrics: Confusion Matrix, Precision, Recall, F1, ROC-AUC</li>
                    <li>11.4 Odds Ratio Analysis: Identifying the Root Cause of Sensor Failure</li>
                </ul>
            </li>
            <li><strong>12. Engineering Synthesis & Fail-Safe ADAS Rules</strong>
                <ul>
                    <li>12.1 Empirical Findings Summary from 50,000 Frames</li>
                    <li>12.2 Practical Fail-Safe Rules for Autonomous Vehicle Control ECUs</li>
                </ul>
            </li>
        </ul>
    </div>
</div>

<!-- CHAPTER 1 -->
<div class="page-break">
    <h1>Chapter 1: The Perception Problem: How Autonomous Vehicles "See"</h1>
    
    <h2>1.1 The Challenge of Self-Driving Cars</h2>
    <p>
        Imagine driving on a dark highway at 100 km/h during torrential rain. Your human eyes struggle with headlight glare, rain sheeting across the windshield, and thick road spray. Now imagine asking a computer to navigate that exact same environment without human intervention. That is the fundamental challenge of <strong>Autonomous Driving</strong>.
    </p>
    <p>
        According to the Society of Automotive Engineers (SAE), autonomy spans from Level 0 (no automation) to Level 5 (full autonomous driving under all conditions without a steering wheel). At <strong>Level 3 (Conditional Automation)</strong> and <strong>Level 4 (High Automation)</strong>, the vehicle's onboard computers—not the human—are legally and physically responsible for driving. If the computer makes a perceptual mistake (such as failing to detect a stalled truck or phantom-braking because of a shadow), lives are immediately at risk.
    </p>

    <div class="callout callout-analogy">
        <div class="callout-title">💡 Real-World Analogy: Human Senses vs. Autonomous Sensors</div>
        A human driver relies primarily on eyes (vision), ears (auditory cues like sirens or honks), and vestibular balance (feeling acceleration and inertia). An autonomous vehicle replaces human eyes with three complementary electronic perception modalities: <strong>Vision Cameras</strong>, <strong>LiDAR</strong>, and <strong>mmWave Radar</strong>.
    </div>

    <h2>1.2 The Perception Triad: Cameras, LiDAR, and Radar</h2>
    <p>
        No single sensor is sufficient to drive an autonomous vehicle safely under all atmospheric and road conditions. Each sensor operates on distinct physical principles with complementary strengths and fatal weaknesses:
    </p>

    <table>
        <thead>
            <tr>
                <th>Perception Modality</th>
                <th>Physical Operating Principle</th>
                <th>Primary Strengths</th>
                <th>Critical Vulnerabilities</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Vision Cameras</strong></td>
                <td>Passive visible light spectrum (400–700 nm). High-resolution RGB pixel grids.</td>
                <td>Superb texture resolution; reads road signs, lane markings, traffic light colors, and pedestrian poses.</td>
                <td>Blinded by direct sun glare, headlight glare, darkness, thick fog, heavy rain, and lens mud/snow cover. Lacks direct 3D depth.</td>
            </tr>
            <tr>
                <td><strong>LiDAR (Light Detection & Ranging)</strong></td>
                <td>Active pulsed infrared laser beams (905 nm or 1550 nm) measuring time-of-flight.</td>
                <td>Produces dense, millimeter-accurate 3D point cloud maps of surrounding objects and obstacles.</td>
                <td>Expensive; suffers severe optical backscatter in heavy rain, dense fog, and snow (laser pulses reflect off water droplets).</td>
            </tr>
            <tr>
                <td><strong>mmWave Radar</strong></td>
                <td>Active radio frequency waves (76–81 GHz, millimeter-scale wavelengths lambda ~ 3.9 mm).</td>
                <td><strong>Penetrates adverse weather</strong> (rain, fog, dust, smoke). Directly measures radial velocity in a single frame via the Doppler effect.</td>
                <td>Lower spatial resolution; susceptible to multipath reflections off guardrails and metallic road clutter.</td>
            </tr>
        </tbody>
    </table>

    <h2>1.3 What is Sensor Fusion?</h2>
    <p>
        <strong>Sensor Fusion</strong> is the algorithmic process of combining sensory data from multiple disparate sources to form a single, coherent, robust understanding of the environment that is superior to what any individual sensor could provide alone.
    </p>
    <p>
        If the camera is blinded by glare, the radar still tracks the vehicle ahead. If the radar experiences ground clutter, LiDAR provides a crisp 3D geometric outline. However, for sensor fusion to work, the central computer must know whether each individual sensor is <strong>healthy</strong> or <strong>degraded</strong>. If a degraded radar feeds corrupted velocity or range readings into the fusion algorithm, the vehicle will make erratic steering or braking decisions. Detecting degradation early is therefore the holy grail of autonomous vehicle safety.
    </p>
</div>

<!-- CHAPTER 2 -->
<div class="page-break">
    <h1>Chapter 2: Radar Physics & Embedded Hardware Architecture</h1>

    <h2>2.1 How Millimeter-Wave (mmWave) FMCW Radar Works</h2>
    <p>
        Automotive radars do not simply send out a single pulse and wait. They use a technique called <strong>Frequency-Modulated Continuous Wave (FMCW)</strong>. The radar transceiver continuously transmits a high-frequency radio signal whose frequency sweeps linearly upward over time—a signal known as a <strong>chirp</strong>.
    </p>
    <p>
        In modern vehicles, this chirp sweeps across the <strong>76 to 81 GHz band</strong> (the center frequency is nominally <strong>77.5 GHz</strong>). Because the wavelength is tiny (lambda = c / f ~ 3.87 mm), the sensor can be packed into a compact housing mounted behind the front bumper or emblem.
    </p>

    <div class="callout callout-formula">
        <div class="callout-title">📐 Physics Formula: Calculating Target Range & Velocity</div>
        <strong>1. Target Range (R):</strong> When the transmitted chirp bounces off an obstacle (e.g., a car ahead) and returns to the radar, it arrives after a round-trip time delay delta_t = 2R / c (where c = 3 x 10^8 m/s is the speed of light). By mixing the received wave with the transmitted wave, a low-frequency <em>beat frequency</em> f_b is produced:
        <div class="math-box">R = (c * T_chirp * f_b) / (2 * Bandwidth)</div>
        <strong>2. Doppler Radial Velocity (v):</strong> If the obstacle is moving relative to the radar, the wave's phase shifts between consecutive chirps due to the <strong>Doppler Effect</strong>:
        <div class="math-box">f_d = (2 * v * f_c) / c  ==>  v = (c * f_d) / (2 * f_c)</div>
        This allows the radar to measure whether an oncoming car is approaching at +25 m/s or moving away at -15 m/s instantaneously in a single telemetry frame!
    </div>

    <h2>2.2 The Embedded Hardware Stack</h2>
    <p>
        Between the physical radio waves bouncing off a truck and the digital bits traveling to the autonomous vehicle brain, a complex chain of electronic hardware operates in real time:
    </p>
    <ol>
        <li><strong>Antenna Array:</strong> Planar patch antennas printed on high-frequency substrate transmit and receive the electromagnetic signals.</li>
        <li><strong>Low-Noise Amplifier (LNA):</strong> The received RF signal is extraordinarily weak (micro-volts). The LNA amplifies this signal while adding as little thermal noise as possible (controlled by Receiver Gain, <code>rx_gain_db</code>).</li>
        <li><strong>Mixer & Analog-to-Digital Converter (ADC):</strong> Mixes the RF signals down to Intermediate Frequencies (IF) and digitizes them into numerical samples.</li>
        <li><strong>Digital Signal Processor (DSP):</strong> Executes 2D and 3D Fast Fourier Transforms (FFTs) across range, Doppler, and azimuth bins to generate target detections.</li>
        <li><strong>Microcontroller Unit (MCU):</strong> The onboard supervisor chip managing radar health, power rails, calibration parameters, and network packaging.</li>
    </ol>

    <h2>2.3 The Vehicle's Nervous System: CAN Bus Communication</h2>
    <p>
        Once the radar calculates target distances, velocities, and angles, it must send this telemetry to the Autonomous Driving Domain Controller (the central GPU/ECU). It does this over the <strong>Controller Area Network (CAN bus)</strong>.
    </p>
    <p>
        The CAN bus is a shared broadcast network. When vehicle subsystems (powertrain, steering, radar, cameras, ABS) all transmit simultaneously, the <strong>CAN bus load percentage</strong> rises. If the bus becomes heavily congested (e.g., >75%), high priority messages may experience queueing delays, increasing the end-to-end <strong>packet latency (ms)</strong>.
    </p>

    <h2>2.4 What is Sensor Degradation?</h2>
    <p>
        A radar sensor does not always fail catastrophically (turning completely off). More dangerously, it often suffers from <strong>gradual degradation</strong>:
    </p>
    <ul>
        <li><strong>Thermal Stress:</strong> Heavy DSP processing inside an uncooled bumper enclosure can push temperatures above 60°C, elevating internal semiconductor noise floors and lowering the Signal-to-Noise Ratio (SNR).</li>
        <li><strong>Calibration Drift:</strong> Road vibrations, thermal expansion, or physical bumps can slightly deform the antenna array or introduce internal phase delays. A high <strong>calibration drift index</strong> means the radar's internal spatial coordinate system is warped, leading to false angle and velocity measurements.</li>
        <li><strong>Supply Voltage Fluctuations:</strong> The MCU operates on a regulated 3.30 V power rail. Electrical transients from vehicle actuators can cause voltage dips or spikes, disrupting ADC conversion precision.</li>
    </ul>
</div>

<!-- CHAPTER 3 -->
<div class="page-break">
    <h1>Chapter 3: Deconstructing the 20 Telemetry Features</h1>
    <p>
        Our dataset (<code>autonomous_radar_telemetry_50k.csv</code>) contains <strong>50,000 observations</strong>. Each row represents a single snapshot frame recorded from an autonomous vehicle radar testbed. Let us break down all 20 columns:
    </p>

    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Feature Name</th>
                <th>Category</th>
                <th>Engineering Unit</th>
                <th>Physical Meaning & Nominal Behavior</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>1</td>
                <td><code>frame_id</code></td>
                <td>Identifier</td>
                <td>String (FRM_XXXXXX)</td>
                <td>Unique sequential identifier for each telemetry snapshot.</td>
            </tr>
            <tr>
                <td>2</td>
                <td><code>vehicle_speed_kmh</code></td>
                <td>Vehicle Context</td>
                <td>km/h</td>
                <td>Speed of the ego vehicle (0.0 to 130.0 km/h). Evenly distributed across city and highway speeds.</td>
            </tr>
            <tr>
                <td>3</td>
                <td><code>radar_frequency_ghz</code></td>
                <td>Radar RF Physics</td>
                <td>GHz</td>
                <td>Operating carrier frequency. Nominally centered at 77.5 GHz with minor oscillator jitter (std dev ~ 0.2 GHz).</td>
            </tr>
            <tr>
                <td>4</td>
                <td><code>tx_power_dbm</code></td>
                <td>Radar RF Physics</td>
                <td>dBm</td>
                <td>RF power transmitted by the antenna. Nominal: ~12.0 dBm (~15.8 mW).</td>
            </tr>
            <tr>
                <td>5</td>
                <td><code>rx_gain_db</code></td>
                <td>Radar RF Physics</td>
                <td>dB</td>
                <td>Amplification factor of the Low-Noise Amplifier in the receiver. Nominal: ~35.0 dB.</td>
            </tr>
            <tr>
                <td>6</td>
                <td><code>snr_db</code></td>
                <td>Radar RF Physics</td>
                <td>dB</td>
                <td><strong>Signal-to-Noise Ratio:</strong> Ratio of true reflected target signal power to background thermal noise. Crucial indicator of measurement confidence.</td>
            </tr>
            <tr>
                <td>7</td>
                <td><code>azimuth_angle_deg</code></td>
                <td>Target Spatial</td>
                <td>Degrees (deg)</td>
                <td>Horizontal angle of detected object relative to vehicle centerline (-45 deg left to +45 deg right).</td>
            </tr>
            <tr>
                <td>8</td>
                <td><code>elevation_angle_deg</code></td>
                <td>Target Spatial</td>
                <td>Degrees (deg)</td>
                <td>Vertical angle of detected object (-10 deg to +10 deg).</td>
            </tr>
            <tr>
                <td>9</td>
                <td><code>target_range_meters</code></td>
                <td>Target Spatial</td>
                <td>Meters (m)</td>
                <td>Distance to the primary obstacle ahead (1.0 m to 200.0 m).</td>
            </tr>
            <tr>
                <td>10</td>
                <td><code>doppler_velocity_mps</code></td>
                <td>Target Spatial</td>
                <td>m/s</td>
                <td>Relative speed of obstacle along the line of sight (negative = approaching, positive = receding).</td>
            </tr>
            <tr>
                <td>11</td>
                <td><code>lidar_point_density</code></td>
                <td>Sensor Fusion</td>
                <td>pts/m²</td>
                <td>Point cloud density returned by auxiliary LiDAR sensor (50 to 2,000 pts/m²).</td>
            </tr>
            <tr>
                <td>12</td>
                <td><code>radar_rcs_dbm2</code></td>
                <td>Radar RF Physics</td>
                <td>dBm²</td>
                <td><strong>Radar Cross Section:</strong> Electromagnetic reflectivity of obstacle (small bicycle ~ 2 dBm², large truck ~ 20 dBm²).</td>
            </tr>
            <tr>
                <td>13</td>
                <td><code>sensor_temp_c</code></td>
                <td>Hardware Health</td>
                <td>°C</td>
                <td>Internal sensor temperature. Nominal: 35–55°C. Critical warning threshold: >62°C.</td>
            </tr>
            <tr>
                <td>14</td>
                <td><code>mcu_voltage_v</code></td>
                <td>Hardware Health</td>
                <td>Volts (V)</td>
                <td>Regulated supply rail voltage for microcontroller. Nominal: 3.300 V +/- 0.04 V.</td>
            </tr>
            <tr>
                <td>15</td>
                <td><code>can_bus_load_pct</code></td>
                <td>Network Health</td>
                <td>%</td>
                <td>Vehicle CAN bus network bandwidth utilization (20% to 85%).</td>
            </tr>
            <tr>
                <td>16</td>
                <td><code>packet_latency_ms</code></td>
                <td>Network Health</td>
                <td>Milliseconds (ms)</td>
                <td>Time taken for radar data packets to traverse CAN bus and reach the central ECU. Nominal: ~15.0 ms.</td>
            </tr>
            <tr>
                <td>17</td>
                <td><code>ambient_lux</code></td>
                <td>Environment</td>
                <td>Lux</td>
                <td>Ambient environmental lighting level (0 lux = pitch dark night, 100,000 lux = blinding direct sunlight).</td>
            </tr>
            <tr>
                <td>18</td>
                <td><code>precipitation_rate_mmh</code></td>
                <td>Environment</td>
                <td>mm/hour</td>
                <td>Atmospheric rainfall rate (0 mm/h = clear dry weather, >8 mm/h = torrential thunderstorm).</td>
            </tr>
            <tr>
                <td>19</td>
                <td><code>calibration_drift_index</code></td>
                <td>Hardware Health</td>
                <td>Unitless [0.0, 1.0]</td>
                <td>Normalized measure of antenna array phase alignment error. 0.0 = factory calibrated, >0.75 = severe hardware misalignment.</td>
            </tr>
            <tr>
                <td>20</td>
                <td><code>sensor_degraded_flag</code></td>
                <td><strong>Target (Label)</strong></td>
                <td>Binary (0 or 1)</td>
                <td><strong>Ground-Truth Health State:</strong> 0 = Sensor is Healthy & Reliable; 1 = Sensor is Degraded / Unsafe.</td>
            </tr>
        </tbody>
    </table>
</div>

<!-- CHAPTER 4 -->
<div class="page-break">
    <h1>Chapter 4: Data Science & Descriptive Statistics from Ground Zero</h1>

    <h2>4.1 What is Tabular Data?</h2>
    <p>
        In data science, we organize information into a <strong>two-dimensional table (DataFrame)</strong>:
    </p>
    <ul>
        <li><strong>Rows (Observations / Records):</strong> Each row represents an individual snapshot in time. In our dataset, we have 50,000 snapshots.</li>
        <li><strong>Columns (Variables / Features):</strong> Each column represents a distinct physical measurement (such as temperature, voltage, or range).</li>
    </ul>

    <h2>4.2 Measures of Central Tendency: Where is the Middle?</h2>
    <p>
        When you have 50,000 numbers for sensor temperature, you cannot read every single one. You need a single number that represents the "typical" value:
    </p>
    <ol>
        <li><strong>The Mean (Average):</strong> Sum of all values divided by count: mean = (sum of X_i) / n.
            <br><em>Vulnerability:</em> The mean is sensitive to extreme outliers. If 99 sensors run at 45°C and 1 sensor catches fire at 500°C, the mean jumps to 49.55°C, giving a misleading impression.
        </li>
        <li><strong>The Median (50th Percentile):</strong> The exact middle number when values are sorted from lowest to highest.
            <br><em>Superpower:</em> The median is completely immune to outliers. In the example above, the median remains exactly 45°C.
        </li>
        <li><strong>The Mode:</strong> The most frequently occurring value in the dataset.</li>
    </ol>

    <h2>4.3 Measures of Dispersion: How Spread Out is the Data?</h2>
    <p>
        Knowing the middle is not enough. Imagine two radar units: both have an average supply voltage of 3.30 V. But Unit A fluctuates between 3.29 V and 3.31 V, while Unit B wild swings between 2.8 V and 3.8 V. Unit B will crash the microcontroller! We quantify spread using:
    </p>
    <ul>
        <li><strong>Variance (s²):</strong> The average squared distance from the mean:
            <div class="math-box">s² = [1 / (n - 1)] * sum( (X_i - mean)² )</div>
        </li>
        <li><strong>Standard Deviation (s):</strong> The square root of variance: s = sqrt(s²).
            <br><em>Why do we use it?</em> Because it is expressed in the <strong>exact same units</strong> as the original measurement! If temperature is in °C, standard deviation is in °C. For our radar temperature, mean = 44.98°C and s = 7.98°C.
        </li>
        <li><strong>Interquartile Range (IQR):</strong> The distance between the 25th percentile (Q1) and 75th percentile (Q3):
            <div class="math-box">IQR = Q3 - Q1</div>
            The IQR captures the middle 50% of your data and is robust against extreme outliers.
        </li>
    </ul>

    <h2>4.4 Shape: Skewness and Kurtosis</h2>
    <ul>
        <li><strong>Skewness:</strong> Measures asymmetry. A symmetric bell curve has skewness ~ 0. Positive skewness (>0) means the distribution has a long tail stretching to the right (like precipitation rate, where most days have little rain but a few have extreme storms). Negative skewness (<0) tails to the left.</li>
        <li><strong>Kurtosis:</strong> Measures the propensity to produce extreme outliers. High kurtosis (>0) means frequent extreme outliers (fat tails). Low kurtosis means thin tails.</li>
    </ul>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_UNIVARIATE__" alt="Univariate Distributions">
        <div class="figure-caption">Figure 4.1: Empirical density plots (histograms + KDE curves) for core physical and system telemetry features from the 50,000-frame dataset. Dashed black lines indicate the Mean; dotted red lines indicate the Median.</div>
    </div>
</div>

<!-- CHAPTER 5 -->
<div class="page-break">
    <h1>Chapter 5: Continuous Probability Distributions & Theoretical Modeling</h1>

    <h2>5.1 What is a Probability Density Function (PDF)?</h2>
    <p>
        In real-world engineering, telemetry measurements are continuous (you can measure 77.49231... GHz). The probability of observing any <em>exact single real number</em> is technically zero (P(X = x) = 0). Instead, we measure the probability that a value falls within an <em>interval</em>:
    </p>
    <div class="math-box">P(a <= X <= b) = Integral from a to b of f(x) dx</div>
    <p>
        The function f(x) is the <strong>Probability Density Function (PDF)</strong>. Fitting theoretical distributions allows engineers to model the likelihood of rare events (e.g., "What is the probability internal temperature exceeds 70°C?").
    </p>

    <h2>5.2 Three Key Distributions in Our Sensor Data</h2>
    <ol>
        <li><strong>The Normal (Gaussian) Distribution N(mu, sigma²):</strong>
            <br>The famous symmetrical bell curve. In physics, whenever an observed value is the result of many tiny, independent additive random noise factors, it naturally forms a Gaussian distribution.
            <br><em>In our data:</em> <code>radar_frequency_ghz</code> follows a Gaussian with mu = 77.501 GHz and sigma = 0.200 GHz.
        </li>
        <li><strong>The Exponential Distribution Exp(lambda):</strong>
            <br>Characterized by a steep decay: values close to zero are very common, but large values become exponentially rarer. It models waiting times between independent Poisson events.
            <br><em>In our data:</em> <code>precipitation_rate_mmh</code> follows an exponential distribution with lambda = 0.671 (mean = 1.49 mm/h). Most driving occurs in light drizzle or dry conditions, but rare storms reach >15 mm/h.
        </li>
        <li><strong>The Continuous Uniform Distribution U(a, b):</strong>
            <br>Every value within the interval [a, b] has equal probability density.
            <br><em>In our data:</em> <code>vehicle_speed_kmh</code> follows U(0, 130 km/h).
        </li>
    </ol>

    <h2>5.3 Goodness-of-Fit Testing: K-S Test & Q-Q Plots</h2>
    <p>
        How do we prove our theoretical distribution actually fits the real data?
    </p>
    <ul>
        <li><strong>Q-Q (Quantile-Quantile) Plot:</strong> We sort our real data and plot its quantiles against theoretical quantiles. If the points fall along a straight diagonal line (y = x), the data perfectly matches the theoretical distribution!</li>
        <li><strong>Kolmogorov-Smirnov (K-S) Test:</strong> Calculates the maximum vertical distance D between the empirical cumulative distribution and the theoretical cumulative distribution. A small D with a large p-value (p > 0.05) mathematically confirms a great fit.</li>
    </ul>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_DIST_FITS__" alt="Fitted Distributions and Q-Q Plots">
        <div class="figure-caption">Figure 5.1: Top row: Empirical histograms overlaid with Maximum Likelihood Estimation (MLE) theoretical PDFs. Bottom row: Quantile-Quantile (Q-Q) plots confirming theoretical distribution fit across physical telemetry.</div>
    </div>
</div>

<!-- CHAPTER 6 -->
<div class="page-break">
    <h1>Chapter 6: Statistical Estimation & The Central Limit Theorem (CLT)</h1>

    <h2>6.1 Population vs. Sample</h2>
    <p>
        In statistics, there is a fundamental distinction between the <strong>Population</strong> and a <strong>Sample</strong>:
    </p>
    <ul>
        <li><strong>The Population (N):</strong> The entire universe of possible observations. In our study, the complete 50,000-frame archive represents the ground-truth population (N = 50,000). The true population mean is denoted by Greek letter mu, and true population standard deviation is sigma.</li>
        <li><strong>The Sample (n):</strong> A small subset of observations drawn randomly from the population (e.g., n = 30 or n = 100). Its sample mean is X_bar and sample standard deviation is s.</li>
    </ul>

    <h2>6.2 The Central Limit Theorem (CLT) Explained Simply</h2>
    <div class="callout callout-formula">
        <div class="callout-title">🧠 The Most Important Theorem in Statistics: The Central Limit Theorem</div>
        If you draw repeated random samples of size n from <em>any</em> population (even if the population is uniform, skewed, or weirdly shaped) and calculate the average X_bar of each sample, the distribution of those sample averages will form a <strong>perfect Gaussian Bell Curve</strong> as n increases (n >= 30).
        <br><br>
        The spread of these sample averages is called the <strong>Standard Error of the Mean (SE)</strong>:
        <div class="math-box">SE = sigma / sqrt(n)</div>
    </div>

    <p>
        Notice that sqrt(n) in the denominator! That square root is why collecting more data improves precision, but with diminishing returns:
    </p>
    <ul>
        <li>To cut your uncertainty (Standard Error) in <strong>half</strong>, you must collect <strong>4 times</strong> as much data (2² = 4).</li>
        <li>To cut uncertainty by a factor of <strong>10</strong>, you must collect <strong>100 times</strong> as much data (10² = 100).</li>
    </ul>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_CLT__" alt="Central Limit Theorem Sampling Distributions">
        <div class="figure-caption">Figure 6.1: Demonstration of the Central Limit Theorem. Sampling distribution of mean SNR across 1,200 repeated trials for sample sizes n = 30, 100, 500, 1000. Notice how the distribution tightens around the true population mean (19.92 dB) as n increases.</div>
    </div>

    <h2>6.3 Constructing Confidence Intervals (CI)</h2>
    <p>
        If you take a sample of n = 100 radar frames and find an average SNR of 20.1 dB, you know the true population mean mu probably isn't <em>exactly</em> 20.1000 dB. How close is it?
    </p>
    <p>
        A <strong>95% Confidence Interval</strong> provides a range:
    </p>
    <div class="math-box">CI_95% = X_bar +/- t_crit * (s / sqrt(n))</div>
    <p>
        <em>Interpretation:</em> If we repeated this experiment 100 times, drawing 100 separate random samples and building an interval from each, approximately 95 of those intervals would contain the true population mean mu.
    </p>
</div>

<!-- CHAPTER 7 -->
<div class="page-break">
    <h1>Chapter 7: Statistical Hypothesis Testing Demystified</h1>

    <h2>7.1 The Scientific Courtroom: How Hypothesis Testing Works</h2>
    <p>
        In everyday life, people argue with opinions. In engineering and science, we make decisions using <strong>Statistical Hypothesis Testing</strong>. Think of hypothesis testing like a criminal courtroom trial:
    </p>
    <div class="callout callout-analogy">
        <div class="callout-title">⚖️ Courtroom Analogy for Hypothesis Testing</div>
        <ul>
            <li><strong>Presumption of Innocence = The Null Hypothesis (H0):</strong> We assume there is "nothing special going on" or "no difference" (innocent until proven guilty).</li>
            <li><strong>The Prosecution's Claim = The Alternative Hypothesis (H1):</strong> The claim we want to prove (e.g., "Degraded sensors run significantly hotter").</li>
            <li><strong>The Evidence = The Test Statistic (t, F, chi², U):</strong> Calculated from sample data.</li>
            <li><strong>Beyond Reasonable Doubt = Significance Level (alpha = 0.05):</strong> If the evidence is so extreme that it had less than a 5% chance of occurring by pure random coincidence, we reject the presumption of innocence (H0) and declare the effect proven!</li>
            <li><strong>The p-Value:</strong> The exact probability of observing our data (or something more extreme) assuming H0 is true. If p < 0.05, we reject H0.</li>
        </ul>
    </div>

    <h2>7.2 Six Formal Hypothesis Tests Executed on Our Telemetry</h2>

    <h3>Test 1: One-Sample t-Test on Radar Center Frequency Compliance</h3>
    <ul>
        <li>H0: mu = 77.50 GHz (The radar operates at standard automotive frequency).</li>
        <li>H1: mu != 77.50 GHz.</li>
        <li><strong>Result:</strong> t = 1.234, p = 0.217. Since p > 0.05, we <strong>Fail to Reject H0</strong>. The radar hardware complies with automotive frequency regulations.</li>
    </ul>

    <h3>Test 2: Two-Sample Independent Welch's t-Test on Sensor Temperature</h3>
    <ul>
        <li>H0: mu_degraded = mu_healthy (Sensor degradation is unrelated to operating temperature).</li>
        <li>H1: mu_degraded > mu_healthy (Degraded sensors suffer thermal stress).</li>
        <li><strong>Result:</strong> t = 2.864, p = 0.0042. Since p < 0.01, we <strong>Reject H0</strong>. Degraded sensors run significantly hotter (+0.26°C on average, p < 0.01).</li>
    </ul>

    <h3>Test 3: Two-Sample Independent t-Test on Received SNR</h3>
    <ul>
        <li>H0: mu_SNR_degraded = mu_SNR_healthy.</li>
        <li>H1: mu_SNR_degraded < mu_SNR_healthy.</li>
        <li><strong>Result:</strong> t = -4.111, p = 3.96 x 10^-5. We <strong>Reject H0</strong>. Degraded sensors exhibit significantly degraded RF signal quality.</li>
    </ul>

    <h3>Test 4: One-Way ANOVA across Speed Regimes</h3>
    <ul>
        <li>H0: mu_latency_low = mu_latency_med = mu_latency_high (Packet communication delay is invariant to road speed).</li>
        <li><strong>Result:</strong> F = 0.14, p = 0.869. <strong>Fail to Reject H0</strong>. Network packet latency is strictly independent of vehicle velocity.</li>
    </ul>

    <h3>Test 5: Non-Parametric Mann-Whitney U Test on Calibration Drift</h3>
    <ul>
        <li>When data distributions are skewed or bounded, we use the Mann-Whitney U rank-sum test.</li>
        <li><strong>Result:</strong> U = 3.8 x 10^8, p < 10^-15. <strong>Reject H0</strong>. Degraded sensors have drastically higher calibration drift.</li>
    </ul>

    <h3>Test 6: Chi-Square (chi²) Test of Independence & Odds Ratios</h3>
    <ul>
        <li>H0: High calibration drift (>0.75) and sensor degradation are independent events.</li>
        <li><strong>Result:</strong> chi² = 38,800.5, p < 10^-15. <strong>Reject H0</strong>.</li>
        <li><strong>Odds Ratio (OR):</strong> OR = 803.9. A sensor with calibration drift >0.75 has an <strong>800-fold higher risk</strong> of being degraded!</li>
    </ul>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_HYPO_TESTS__" alt="Hypothesis Testing Boxplots and Heatmap">
        <div class="figure-caption">Figure 7.1: Visual validation of hypothesis tests. Left: Sensor temperature comparison between healthy and degraded states (p = 0.0042). Center: Received SNR comparison (p < 0.0001). Right: Contingency table heatmap showing strong dependence between calibration drift trigger and degradation (chi² = 38,800.5).</div>
    </div>
</div>

<!-- CHAPTER 8 -->
<div class="page-break">
    <h1>Chapter 8: Correlation, Regression & Radar Path Loss Physics</h1>

    <h2>8.1 Correlation vs. Causation</h2>
    <p>
        When two variables move together, they are <strong>correlated</strong>:
    </p>
    <ul>
        <li><strong>Pearson Correlation (r):</strong> Measures linear relationships. Values range from -1.0 (perfect negative line) to +1.0 (perfect positive line). r = 0 means zero linear association.</li>
        <li><strong>Spearman Rank Correlation (rho):</strong> Measures monotonic relationships (whether Y increases when X increases, regardless of whether the line is straight or curved). Robust against outliers.</li>
    </ul>
    <div class="callout callout-warning">
        <div class="callout-title">⚠️ Golden Rule of Data Science: Correlation != Causation</div>
        Just because variable A correlates with variable B does not mean A causes B. For instance, CAN bus load and ambient sunlight might have a slight random sample correlation, but ambient sunlight does not cause CAN bus bytes to transmit! Always inspect the physical mechanism.
    </div>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_CORR_HEATMAP__" alt="Correlation Heatmap">
        <div class="figure-caption">Figure 8.1: Full Pearson correlation matrix heatmap across physical telemetry, embedded electronics, and the sensor degradation flag.</div>
    </div>

    <h2>8.2 Simple Linear Regression: Modeling Radar Path Loss</h2>
    <p>
        In physics, as radio waves travel away from an antenna, spread out spherically, bounce off a target, and return, their power attenuates with distance. We model this using <strong>Ordinary Least Squares (OLS) Linear Regression</strong>:
    </p>
    <div class="math-box">Y = beta_0 + beta_1 * X + epsilon</div>
    <p>
        Fitting <code>snr_db</code> as a function of <code>target_range_meters</code> yields:
    </p>
    <div class="math-box">SNR (dB) = 29.98 - 0.0998 * Range (meters)</div>
    <ul>
        <li><strong>Intercept (beta_0 = 29.98 dB):</strong> The theoretical maximum SNR when an obstacle is right in front of the bumper (Range ~ 0 m).</li>
        <li><strong>Slope (beta_1 = -0.0998 dB/m):</strong> For every 10 meters of distance to an obstacle, received SNR drops by approximately <strong>1.0 dB</strong>!</li>
        <li><strong>Coefficient of Determination (R² = 0.891):</strong> Target distance alone explains <strong>89.1%</strong> of all variation in radar signal strength!</li>
    </ul>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_BIVARIATE__" alt="Bivariate Relationships">
        <div class="figure-caption">Figure 8.2: Bivariate engineering relationships. Top-left: Linear path loss of SNR over target distance (R² = 0.891). Top-right: Latency vs. CAN bus load. Bottom-left: Temperature vs. calibration drift colored by degradation state. Bottom-right: Precipitation vs. Radar RCS.</div>
    </div>

    <h2>8.3 Regression Residual Diagnostics</h2>
    <p>
        For a regression model to be statistically valid, its residuals (prediction errors epsilon = Y - Y_hat) must satisfy the Gauss-Markov assumptions:
    </p>
    <ol>
        <li><strong>Zero Mean:</strong> E[epsilon] = 0.</li>
        <li><strong>Homoscedasticity:</strong> The spread of residuals must be constant across all predicted values (no funnel shape).</li>
        <li><strong>Normality:</strong> The residuals must follow a Gaussian bell curve (verified via Normal Q-Q plot).</li>
    </ol>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_RESIDUALS__" alt="Residual Diagnostics">
        <div class="figure-caption">Figure 8.3: Residual diagnostics for latency regression. Left: Residuals vs. Fitted values showing uniform spread (homoscedasticity). Right: Normal Q-Q plot confirming Gaussian error distribution.</div>
    </div>
</div>

<!-- CHAPTER 9 -->
<div class="page-break">
    <h1>Chapter 9: Electronic Anomaly Detection: Z-Score vs. IQR</h1>

    <h2>9.1 Why Do We Need Anomaly Detection?</h2>
    <p>
        In an autonomous vehicle, a single corrupted sensor reading or a transient voltage brownout can trigger dangerous emergency maneuvers. Anomaly detection algorithms constantly monitor incoming telemetry to catch out-of-spec physical behavior before it causes accidents.
    </p>

    <h2>9.2 Method 1: The Parametric Z-Score Rule (|Z| > 3.0)</h2>
    <p>
        The Z-score measures how many standard deviations an observation is away from the mean:
    </p>
    <div class="math-box">Z = (X - mu) / sigma</div>
    <ul>
        <li>If data is Gaussian, the <strong>Empirical 68-95-99.7 Rule</strong> tells us that 99.73% of all observations lie within +/- 3-sigma.</li>
        <li>Any point with |Z| > 3.0 has less than a 0.27% probability of occurring under normal conditions and is flagged as an anomaly.</li>
    </ul>

    <h2>9.3 Method 2: The Non-Parametric IQR Fence Rule</h2>
    <p>
        If a feature is skewed or has heavy tails, the mean and standard deviation become distorted by the very outliers we want to find. John Tukey invented the <strong>Interquartile Range (IQR) Rule</strong>:
    </p>
    <div class="math-box">Lower Fence = Q1 - 1.5 * IQR,     Upper Fence = Q3 + 1.5 * IQR</div>
    <p>
        Any observation falling outside these fences is flagged as an outlier.
    </p>

    <table>
        <thead>
            <tr>
                <th>Electronic Variable</th>
                <th>Nominal Mean</th>
                <th>Z-Score Outliers (|Z|>3)</th>
                <th>Z %</th>
                <th>IQR Outliers (1.5xIQR)</th>
                <th>IQR %</th>
                <th>Acceptable Normal Range</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>mcu_voltage_v</code></td>
                <td>3.300 V</td>
                <td>134 frames</td>
                <td>0.27%</td>
                <td>352 frames</td>
                <td>0.70%</td>
                <td>[3.219 V, 3.381 V]</td>
            </tr>
            <tr>
                <td><code>sensor_temp_c</code></td>
                <td>44.98 °C</td>
                <td>138 frames</td>
                <td>0.28%</td>
                <td>337 frames</td>
                <td>0.67%</td>
                <td>[28.84 °C, 61.12 °C]</td>
            </tr>
            <tr>
                <td><code>packet_latency_ms</code></td>
                <td>14.99 ms</td>
                <td>134 frames</td>
                <td>0.27%</td>
                <td>346 frames</td>
                <td>0.69%</td>
                <td>[8.91 ms, 21.07 ms]</td>
            </tr>
        </tbody>
    </table>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_ANOMALIES__" alt="Anomaly Detection Plots">
        <div class="figure-caption">Figure 9.1: Anomaly detection visualizations. Left: Bivariate scatter of MCU Voltage vs. Sensor Temperature, with |Z| > 3.0 electronic anomalies highlighted in red. Right: Standardized boxplots showing outlier bounds.</div>
    </div>
</div>

<!-- CHAPTER 10 -->
<div class="page-break">
    <h1>Chapter 10: Sequential Telemetry & Time-Series Dynamics</h1>

    <h2>10.1 Tracking Sensor Signals Across Consecutive Frames</h2>
    <p>
        Autonomous vehicles do not analyze telemetry in isolated vacuum snapshots; radar data arrives continuously at 20–50 Hz (every 20 to 50 milliseconds).
    </p>
    <p>
        To filter high-frequency electronic noise and track baseline drift, we calculate <strong>Rolling Moving Averages</strong> over a window of W = 50 frames:
    </p>
    <div class="math-box">MA_t = (1 / W) * sum_{i=0}^{W-1} X_{t-i}</div>
    <p>
        By adding dynamic volatility envelopes (MA_t +/- 2 * sigma_t), autonomous vehicle software can detect sudden thermal spikes or signal dropouts that breach normal operating envelopes.
    </p>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_TIME_SERIES__" alt="Sequential Telemetry Profile">
        <div class="figure-caption">Figure 10.1: Sequential radar telemetry over 1,000 operational frames. Top: Sensor temperature with rolling moving average (red) and +/- 2-sigma volatility band. Bottom: Received SNR profile across consecutive frames.</div>
    </div>

    <h2>10.2 Autocorrelation Function (ACF)</h2>
    <p>
        The <strong>Autocorrelation Function (ACF)</strong> measures the correlation between a signal and a time-lagged copy of itself:
    </p>
    <div class="math-box">r_k = [ sum( (X_t - mean)*(X_{t+k} - mean) ) ] / [ sum( (X_t - mean)² ) ]</div>
    <p>
        In stationary radar operating conditions, autocorrelation drops rapidly to near zero for lags k >= 1, proving that consecutive frames represent independent thermal and Doppler measurements.
    </p>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_ACF__" alt="Autocorrelation Function">
        <div class="figure-caption">Figure 10.2: Autocorrelation plots (ACF) up to lag 40 for sensor temperature and vehicle speed, showing rapid decay into the 95% confidence bounds (blue shaded area).</div>
    </div>
</div>

<!-- CHAPTER 11 -->
<div class="page-break">
    <h1>Chapter 11: Predictive Machine Learning for Early Sensor Degradation</h1>

    <h2>11.1 The Machine Learning Workflow</h2>
    <p>
        The ultimate goal of our project (from the infographic schematic) is:
        <br><em>"By analysing these signals and their patterns, we predict whether the sensor is healthy or degraded — enabling early fault detection and safer autonomous driving."</em>
    </p>
    <p>
        This is a <strong>Supervised Binary Classification</strong> problem:
    </p>
    <ul>
        <li><strong>Input Features (X):</strong> 16 physical, hardware, network, and environmental signals.</li>
        <li><strong>Target (Y):</strong> <code>sensor_degraded_flag</code> (0 = Healthy, 1 = Degraded).</li>
    </ul>

    <h2>11.2 Logistic Regression & The Sigmoid Function</h2>
    <p>
        Standard linear regression predicts values from -infinity to +infinity, which doesn't make sense for a probability (probabilities must lie strictly between 0 and 1).
        <strong>Logistic Regression</strong> maps any real-valued number into a valid probability using the <strong>Sigmoid Function</strong>:
    </p>
    <div class="math-box">P(Y = 1 | X) = 1 / [1 + e^(-z)],     where z = beta_0 + sum( beta_j * X_j )</div>

    <h2>11.3 Evaluating Model Performance</h2>
    <p>
        We train on 40,000 frames (80%) and test on 10,000 unseen frames (20%). We evaluate using:
    </p>
    <ul>
        <li><strong>Confusion Matrix:</strong>
            <ul>
                <li><strong>True Negative (TN = 7,813):</strong> Correctly predicted healthy sensors.</li>
                <li><strong>True Positive (TP = 2,167):</strong> Correctly caught degraded sensors!</li>
                <li><strong>False Positive (FP = 20):</strong> False alarm (predicted degraded when actually healthy).</li>
                <li><strong>False Negative (FN = 0):</strong> Critical miss! (The model never missed a single degraded sensor!).</li>
            </ul>
        </li>
        <li><strong>Accuracy:</strong> (TP + TN) / Total = 99.80%.</li>
        <li><strong>Precision:</strong> TP / (TP + FP) = 99.09%.</li>
        <li><strong>Recall (Sensitivity):</strong> TP / (TP + FN) = 100.0%.</li>
        <li><strong>ROC-AUC (Area Under the Curve):</strong> 0.9998 (Near-perfect discrimination).</li>
    </ul>

    <div class="figure-container">
        <img src="data:image/png;base64,__IMG_ML_EVAL__" alt="Machine Learning Evaluation">
        <div class="figure-caption">Figure 11.1: Predictive model evaluation. Left: Confusion Matrix showing 99.8% classification accuracy on unseen test telemetry. Right: Receiver Operating Characteristic (ROC) curve with near-perfect AUC = 0.9998.</div>
    </div>

    <h2>11.4 Odds Ratios: What Triggers Sensor Degradation?</h2>
    <p>
        By taking the exponential of the model's standardized coefficients (e^beta_j), we calculate the <strong>Odds Ratio</strong> for each physical feature. This reveals the root causes of failure:
    </p>
    <ol>
        <li><strong><code>calibration_drift_index</code> (Standardized beta = +7.42, Odds Ratio = 1,668):</strong> The single most dominant failure driver. Physical array misalignment exponentially increases degradation probability.</li>
        <li><strong><code>snr_db</code> (Standardized beta = -2.18, Odds Ratio = 0.113):</strong> Strong protective factor; high SNR guarantees healthy operation. When SNR collapses below 8 dB, failure risk spikes.</li>
        <li><strong><code>sensor_temp_c</code> (Standardized beta = +1.15, Odds Ratio = 3.16):</strong> Every 1-sigma increase in temperature triples the odds of degradation.</li>
        <li><strong><code>precipitation_rate_mmh</code> (Standardized beta = +0.58, Odds Ratio = 1.79):</strong> Heavy rainfall acts as an environmental catalyst compounding existing hardware degradation.</li>
    </ol>
</div>

<!-- CHAPTER 12 -->
<div class="page-break">
    <h1>Chapter 12: Engineering Synthesis & Fail-Safe ADAS Rules</h1>

    <h2>12.1 Executive Summary of Empirical Findings</h2>
    <p>
        Through 50,000 frames of radar and sensor fusion telemetry, our statistical analysis has demonstrated:
    </p>
    <ul>
        <li><strong>Radar Physical Resilience:</strong> mmWave radar operating at 77.5 GHz exhibits exceptional stability across variable ambient lighting (0 to 100,000 lux) and vehicle speeds (0 to 130 km/h). Its path loss attenuation adheres to a deterministic linear decay of -0.10 dB/m (R² = 0.891).</li>
        <li><strong>The Critical Failure Trigger:</strong> Sensor degradation is not a random black-swan event. It is driven by the trifecta of <strong>calibration drift (>0.75)</strong>, <strong>thermal overheating (>62°C)</strong>, and <strong>RF signal collapse (SNR < 8 dB)</strong>.</li>
        <li><strong>Electronic Stability:</strong> Automotive power rails show tight Gaussian regulation (3.300 +/- 0.040 V), with electronic anomalies occurring in less than 0.3% of frames.</li>
    </ul>

    <h2>12.2 Practical Fail-Safe Rules for Autonomous Vehicle Control ECUs</h2>
    <p>
        Based on our empirical findings, we propose four concrete software rules for the Autonomous Driving Domain Controller:
    </p>

    <div class="callout callout-warning">
        <div class="callout-title">🛑 ADAS Fail-Safe Rule 1: Immediate Sensor Demotion</div>
        If incoming radar telemetry reports <code>calibration_drift_index > 0.75</code> OR <code>snr_db < 8.0 dB</code> while target distance is <50 m, the central ECU must immediately demote the radar from <em>Primary Autonomous Guidance</em> to <em>Degraded Advisory Mode</em>, handover steering to camera/LiDAR fusion, and alert the driver.
    </div>

    <div class="callout callout-formula">
        <div class="callout-title">❄️ ADAS Fail-Safe Rule 2: Active Thermal Throttling</div>
        If <code>sensor_temp_c</code> exceeds 60.0°C for more than 100 consecutive frames, activate auxiliary cooling fans or reduce the FMCW chirp repetition frequency to prevent thermal degradation cascades.
    </div>

    <div class="callout callout-analogy">
        <div class="callout-title">🌧️ ADAS Fail-Safe Rule 3: Adverse Weather Weight Adaptation</div>
        When <code>precipitation_rate_mmh > 5.0 mm/h</code>, increase the fusion confidence weight of radar Doppler velocity by +35% while reducing the weight of optical LiDAR point clouds, which suffer severe backscatter attenuation in rain.
    </div>

    <div class="callout callout-title">
        <div class="callout-title" style="color: #0369a1;">🔧 ADAS Fail-Safe Rule 4: Predictive Service Scheduling</div>
        If a vehicle's 500-frame rolling average calibration drift exceeds 0.60, schedule an over-the-air (OTA) calibration alert or dealer service notification before the sensor reaches the critical 0.75 failure boundary.
    </div>

    <br>
    <div style="text-align: center; margin-top: 30px; font-weight: 700; color: #1e3a8a; font-size: 11pt;">
        — END OF HANDBOOK —<br>
        <span style="font-size: 9pt; color: #64748b; font-weight: 400;">Statistical Data Analytics | Autonomous Vehicle Radar & Sensor Fusion Telemetry Investigation</span>
    </div>
</div>

</body>
</html>
"""

    html = template
    html = html.replace("__IMG_TARGET_DIST__", img_target_dist)
    html = html.replace("__IMG_UNIVARIATE__", img_univariate)
    html = html.replace("__IMG_BIVARIATE__", img_bivariate)
    html = html.replace("__IMG_CORR_HEATMAP__", img_corr_heatmap)
    html = html.replace("__IMG_DIST_FITS__", img_dist_fits)
    html = html.replace("__IMG_CLT__", img_clt)
    html = html.replace("__IMG_HYPO_TESTS__", img_hypo_tests)
    html = html.replace("__IMG_RESIDUALS__", img_residuals)
    html = html.replace("__IMG_ANOMALIES__", img_anomalies)
    html = html.replace("__IMG_TIME_SERIES__", img_time_series)
    html = html.replace("__IMG_ACF__", img_acf)
    html = html.replace("__IMG_ML_EVAL__", img_ml_eval)

    return html

if __name__ == '__main__':
    nb_path = r'd:\SDA\autonomous_radar_telemetry_analysis.ipynb'
    html_path = r'd:\SDA\handbook.html'
    pdf_path = r'd:\SDA\Autonomous_Radar_Sensor_Fusion_and_Data_Analytics_Handbook.pdf'

    print("Extracting high-resolution base64 plots from notebook...")
    images = extract_notebook_images(nb_path)
    print(f"Extracted {len(images)} plot images.")

    print("Generating comprehensive educational HTML textbook...")
    html_content = build_handbook_html(images)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"HTML textbook written to {html_path} ({len(html_content):,} bytes).")

    print("Converting HTML to publication-grade PDF via Microsoft Edge headless...")
    edge_exe = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
    cmd = [
        edge_exe,
        '--headless',
        '--disable-gpu',
        '--run-all-compositor-stages-before-draw',
        f'--print-to-pdf={pdf_path}',
        html_path
    ]
    subprocess.run(cmd, check=True)

    time.sleep(2)
    if os.path.exists(pdf_path):
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"SUCCESS: Generated PDF handbook at: {pdf_path} ({size_mb:.2f} MB)")
    else:
        print("ERROR: PDF was not found after execution.")
