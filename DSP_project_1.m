    % =========================================================================
% STEP 1: LOAD AND INSPECT PADERBORN DATASET
% =========================================================================
clear; clc; close all;
load N09_M07_F10_KA01_1.mat
% 1. Load the file
filename = 'N09_M07_F10_KA01_1.mat'; % <--- Change to your actual file name
fprintf('Loading %s...\n', filename);
data_struct = load(filename);

% Paderborn files save as a top-level struct matching the file name
struct_field = fieldnames(data_struct);
raw_data = data_struct.(struct_field{1});

% 2. Extract Sampling Frequency and Vibration Channel
% Channel 7 in Paderborn format is typically the 64 kHz High-Speed Vibration Sensor
fs = 64000; % 64 kHz
vibration_raw = raw_data.Y(7).Data; % Raw acceleration values

% 3. Extract 1 Second Frame (64,000 samples)
N = fs; % 1 second window
frame = vibration_raw(1:N);
t = (0:N-1) / fs;

% 4. Compute Baseline DSP Features
frame_rms = rms(frame);
frame_kurtosis = kurtosis(frame);

fprintf('--- FRAME FEATURE SUMMARY ---\n');
fprintf('Sampling Frequency: %d Hz\n', fs);
fprintf('Total Frame Samples: %d\n', length(frame));
fprintf('Time-Domain RMS:    %.4f g\n', frame_rms);
fprintf('Time-Domain Kurtosis: %.4f\n', frame_kurtosis);

% 5. Plot Time-Domain Waveform
figure('Name', 'Raw Vibration Signal - 1 Second Frame');
plot(t, frame, 'Color', [0 0.4470 0.7410]);
grid on;
xlabel('Time (seconds)');
ylabel('Acceleration (g)');
title(sprintf('Raw Vibration Signal (RMS = %.3f, Kurtosis = %.3f)', frame_rms, frame_kurtosis));
% =========================================================================
% STEP 2: SPECTRAL KURTOSIS & ENVELOPE SPECTRUM ANALYSIS
% =========================================================================

% Run spectral kurtosis once
[Kmax, Level, fc, bw] = kurtogram(frame, fs);
fc = fc(1);
bw = bw(1);


fprintf('--- SPECTRAL KURTOSIS OPTIMIZATION ---\n');
fprintf('Max Kurtosis in Sub-band: %.4f\n', Kmax);
fprintf('Optimal Center Frequency (fc): %.2f Hz\n', fc);
fprintf('Optimal Bandwidth (bw):        %.2f Hz\n', bw);

% Force scalar, valid passband limits
f_lower = fc - bw/2;
f_upper = fc + bw/2;

f_lower = max(f_lower, 1);
f_upper = min(f_upper, fs/2 - 1);

if ~isscalar(f_lower) || ~isscalar(f_upper) || f_upper <= f_lower
    f_lower = max(1, fc - 50);
    f_upper = min(fs/2 - 1, fc + 50);
end

fpass = [f_lower f_upper];
filtered_frame = bandpass(frame(:), fpass, fs);

% 3. Envelope Demodulation via Hilbert Transform
analytic_signal = hilbert(filtered_frame);
envelope_signal = abs(analytic_signal);
% =========================================================================
% =========================================================================
% STEP 2: SPECTRAL KURTOSIS & ENVELOPE SPECTRUM ANALYSIS
% =========================================================================

% 1. Run Spectral Kurtosis to automatically find optimal filter band
[kgram, ~, ~, fc, ~, bw] = kurtogram(filtered_frame, fs);
Kmax = max(kgram(:));

fprintf('--- SPECTRAL KURTOSIS OPTIMIZATION ---\n');
fprintf('Max Kurtosis in Kurtogram: %.4f\n', Kmax);
fprintf('Optimal Center Frequency (fc): %.2f Hz\n', fc);
fprintf('Optimal Bandwidth (bw):        %.2f Hz\n', bw);

% 2. Apply Bandpass Filter around the resonance center frequency
f_lower = max(100, fc - bw/2);
f_upper = min(fs/2 - 100, fc + bw/2);

if f_upper <= f_lower
    f_lower = max(1, fc - 50);
    f_upper = min(fs/2 - 1, fc + 50);
end

filtered_frame = bandpass(frame, [f_lower, f_upper], fs);

% 3. Envelope Demodulation via Hilbert Transform
analytic_signal = hilbert(filtered_frame);
envelope_signal = abs(analytic_signal);

% Subtract DC offset for clean FFT
envelope_detrended = envelope_signal - mean(envelope_signal);

% 4. Compute FFT of the Envelope Signal
N_env = length(envelope_detrended);
env_fft = abs(fft(envelope_detrended)) / N_env;
env_fft = env_fft(1:floor(N_env/2)+1);
env_fft(2:end-1) = 2 * env_fft(2:end-1);
f_axis = (0:floor(N_env/2)) * (fs / N_env);

% 5. Plot Envelope Spectrum (Focusing on 0 to 500 Hz Fault Zone)
figure('Name', 'Envelope Spectrum (Demodulated Signal)');
plot(f_axis, env_fft, 'r', 'LineWidth', 1.2);
grid on;
xlim([0 500]);
xlabel('Frequency (Hz)');
ylabel('Envelope Amplitude');
title(sprintf('Envelope Spectrum (Filtered at fc = %.0f Hz, bw = %.0f Hz)', fc, bw));

% 6. Find Dominant Peak Frequency in the Fault Band (10 Hz to 500 Hz)
[~, idx_start] = min(abs(f_axis - 10));
[~, idx_end] = min(abs(f_axis - 500));
[~, max_idx] = max(env_fft(idx_start:idx_end));
dominant_freq = f_axis(idx_start + max_idx - 1);

fprintf('\nDominant Fault Peak in Envelope Spectrum: %.2f Hz\n', dominant_freq);

% =========================================================================
% STEP 3: SLIDING WINDOW FEATURE EXTRACTION (HEALTH INDICATOR MATRIX)
% =========================================================================

window_len = 0.1 * fs; % 100 ms window (6,400 samples)
num_windows = floor(length(vibration_raw) / window_len);

% Preallocate Feature Matrix [RMS, Kurtosis, Crest Factor, Envelope Mean]
feature_matrix = zeros(num_windows, 4);

for i = 1:num_windows
    idx_start = (i-1)*window_len + 1;
    idx_end = i*window_len;
    sub_frame = vibration_raw(idx_start:idx_end);

    % Time-domain metrics
    w_rms = rms(sub_frame);
    w_kurt = kurtosis(sub_frame);
    w_crest = max(abs(sub_frame)) / w_rms;

    % Envelope energy metric
    sub_filtered = bandpass(sub_frame, fpass, fs);
    sub_env = abs(hilbert(sub_filtered));
    w_env_mean = mean(sub_env);

    feature_matrix(i, :) = [w_rms, w_kurt, w_crest, w_env_mean];
end

% Plot Feature Trajectories Over Time
figure('Name', 'Extracted Health Indicators Over Time');
subplot(2,1,1);
plot(feature_matrix(:, 2), 'k-o', 'LineWidth', 1.2);
ylabel('Kurtosis'); grid on;
title('Time-Series Feature Extraction (Health Indicators)');

subplot(2,1,2);
plot(feature_matrix(:, 4), 'r-s', 'LineWidth', 1.2);
xlabel('Window Index'); ylabel('Envelope Energy'); grid on;

% =========================================================================
% OPTIMIZED STEP 4: CORRELATION-FILTERED PCA FUSION
% =========================================================================

% 1. Select energy-dense co-varying features [RMS, Envelope Mean]
refined_features = feature_matrix(:, [1, 4]); % Drop Kurtosis (col 2) & Crest Factor (col 3)

% 2. Standardize features (Z-score normalization)
mu = mean(refined_features);
sigma = std(refined_features);
norm_features = (refined_features - mu) ./ sigma;

% 3. Principal Component Analysis (PCA)
[coeff, score, ~, ~, explained] = pca(norm_features);
pc1 = score(:, 1); 

% Align health index decay direction
if corr(pc1, (1:length(pc1))') > 0
    pc1 = -pc1;
end

% 4. Normalize Health Index (1.0 = Healthy, 0.0 = Degraded)
HI = (pc1 - min(pc1)) / (max(pc1) - min(pc1));

% 5. Fit Exponential Model
t_axis = (1:length(HI))';
log_HI = log(max(HI, 0.01)); 
p = polyfit(t_axis, log_HI, 1);
b_fit = p(1);
a_fit = exp(p(2));
hi_model = a_fit * exp(b_fit * t_axis);

fprintf('\n--- REFINED PROGNOSTIC SUMMARY ---\n');
fprintf('Optimized PC1 Variance Explained: %.2f%%\n', explained(1));
fprintf('Decay Slope (b):                   %.4f\n', b_fit);

% =========================================================================
% TWO-STATE HEALTH INDEX CALIBRATION (DYNAMIC FRAME INDEXING)
% =========================================================================
    

files = {'N09_M07_F10_K001_1.mat', 'N09_M07_F10_KA01_1.mat'};
fs = 64000;
Fpass = [26000, 28000];

feature_matrix = [];
frame_counts = zeros(length(files), 1);

for k = 1:length(files)
    clear data_struct raw_sig;
    data_struct = load(files{k});
    fnames = fieldnames(data_struct);
    raw_sig = data_struct.(fnames{1}).Y(7).Data;
    
    num_frames = floor(length(raw_sig) / fs);
    frame_counts(k) = num_frames;
    
    for n = 1:num_frames
        idx = (n-1)*fs + 1 : n*fs;
        frame = raw_sig(idx);
        
        f_rms = rms(frame);
        filt_sig = bandpass(frame, Fpass, fs);
        f_env = mean(abs(hilbert(filt_sig)));
        
        feature_matrix = [feature_matrix; f_rms, f_env];
    end
end

% Standardize features
norm_features = (feature_matrix - mean(feature_matrix)) ./ std(feature_matrix);
[~, score, ~, ~, explained] = pca(norm_features);

pc1 = score(:, 1);
if corr(pc1, (1:length(pc1))') > 0
    pc1 = -pc1;
end

% Normalize Health Index (1.0 = Healthy, 0.0 = Damaged)
HI = (pc1 - min(pc1)) / (max(pc1) - min(pc1));

% Dynamic split indices
idx_k001 = 1 : frame_counts(1);
idx_ka01 = (frame_counts(1) + 1) : sum(frame_counts);

fprintf('--- DYNAMIC CALIBRATION RESULTS ---\n');
fprintf('Total Frames Processed: %d (%d K001, %d KA01)\n', length(HI), frame_counts(1), frame_counts(2));
fprintf('PC1 Variance Explained: %.2f%%\n', explained(1));
fprintf('Average HI for Healthy (K001): %.4f\n', mean(HI(idx_k001)));
fprintf('Average HI for Damaged (KA01): %.4f\n', mean(HI(idx_ka01)));

% Plot Step Shift
figure('Name', 'Calibrated Health Index Shift');
plot(idx_k001, HI(idx_k001), 'g-o', 'LineWidth', 1.5, 'DisplayName', 'K001 (Healthy)'); hold on;
plot(idx_ka01, HI(idx_ka01), 'r-s', 'LineWidth', 1.5, 'DisplayName', 'KA01 (Outer Race Fault)');
grid on; ylabel('Health Index (HI)'); xlabel('Total Frame Index');
title('Calibrated Health Index Shift (Healthy vs Damaged)');
legend('Location', 'northeast');