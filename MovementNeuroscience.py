import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

def load_mat(filepath):
    """Load a .mat file, trying scipy first, then h5py for v7.3 files."""
    try:
        data = sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
        return data
    except NotImplementedError:
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "This .mat file is in v7.3 (HDF5) format. "
                "Install h5py to read it: pip install h5py"
            )
        return _load_mat_h5(filepath)


def _load_mat_h5(filepath):
    """
    Load a MATLAB v7.3 (.mat) file using h5py.
    Handles cell arrays (stored as HDF5 object references) and regular arrays.
    """
    import h5py

    def _deref(item, f):
        """Recursively dereference HDF5 object references."""
        if isinstance(item, h5py.Reference):
            return _deref(f[item], f)
        if isinstance(item, h5py.Dataset):
            raw = item[()]
            # Cell array: array of object references
            if raw.dtype == object:
                out = np.empty(raw.shape, dtype=object)
                it = np.nditer(raw, flags=['multi_index', 'refs_ok'])
                while not it.finished:
                    out[it.multi_index] = _deref(raw[it.multi_index], f)
                    it.iternext()
                return out
            return raw
        if isinstance(item, h5py.Group):
            return {k: _deref(v, f) for k, v in item.items()}
        return item

    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):  # skip internal HDF5 metadata
                continue
            data[key] = _deref(f[key], f)
    return data

# Spike-triggered averaging

def spike_triggered_averaging(EMGSig, MUPulses_sorted, window_sec, fsamp):
    """
    Compute spike-triggered average (STA) for each motor unit.

    Parameters
    ----------
    EMGSig : ndarray, shape (n_samples, 16)  OR  (16, n_samples)
        Raw EMG signals for 16 channels.
    MUPulses_sorted : list of 1-D arrays
        Spike sample indices for each MU (sorted by recruitment order).
    window_sec : float
        Half-window size in seconds (total window = 2 * window_sec).
    fsamp : float
        Sampling frequency in Hz.

    Returns
    -------
    STA_mean : list of (8, 2) cell-like lists
        STA_mean[n][row][col] is the averaged MUAP waveform for MU n,
        channel (row, col) with col in {0,1} and row in 0..7.
    """
    half_win = int(window_sec * fsamp / 2)
    win_len = 2 * half_win + 1

    # Ensure EMGSig shape is (n_samples, 16)
    if EMGSig.shape[0] == 16:
        EMGSig = EMGSig.T

    n_samples = EMGSig.shape[0]
    n_channels = EMGSig.shape[1]  # should be 16

    STA_mean = []

    for spikes in MUPulses_sorted:
        spikes = np.asarray(spikes, dtype=int) - 1  # MATLAB 1-based → 0-based

        # Accumulate snippets for each channel
        channel_sum = np.zeros((win_len, n_channels))
        count = 0

        for s in spikes:
            start = s - half_win
            end = s + half_win + 1
            if start >= 0 and end <= n_samples:
                channel_sum += EMGSig[start:end, :]
                count += 1

        if count > 0:
            channel_avg = channel_sum / count
        else:
            channel_avg = channel_sum

        # Reshape into 8×2 cell equivalent  (columns: E1-E8, E9-E16)
        muap_grid = [[None] * 2 for _ in range(8)]
        for col in range(2):
            for row in range(8):
                ch_idx = col * 8 + row
                if ch_idx < n_channels:
                    muap_grid[row][col] = channel_avg[:, ch_idx]

        STA_mean.append(muap_grid)

    return STA_mean

# Load data

print("Loading data...")
mat_data = load_mat('iEMG_contraction.mat')

# Robust extraction — handles both scipy (old .mat) and h5py (v7.3) layouts

def _extract_mu_pulses(raw):
    """
    Convert MUPulses from whatever format load_mat returned into a plain
    Python list of 1-D integer numpy arrays (1-based sample indices).
    Handles: scipy cell-array, h5py object-array of references/arrays.
    """
    pulses_list = []
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        flat = raw.flatten()
        for item in flat:
            arr = np.asarray(item, dtype=float).flatten()
            pulses_list.append(arr.astype(int))
    else:
        for item in list(raw):
            arr = np.asarray(item, dtype=float).flatten()
            pulses_list.append(arr.astype(int))
    return pulses_list


MUPulses = _extract_mu_pulses(mat_data['MUPulses'])
force_signal = np.asarray(mat_data['force_signal'], dtype=float).flatten()
fsamp = float(np.asarray(mat_data['fsamp']).flat[0])

# --- Robust EMGSig extraction for h5py v7.3 layout ---
# In v7.3, EMGSig may be stored as a Group of per-channel datasets,
# or as a 2-D dataset that is transposed relative to MATLAB convention.

raw_emg = mat_data['EMGSig']

if isinstance(raw_emg, dict):
    # Stored as a group: keys are channel names / indices
    channels = [np.asarray(raw_emg[k], dtype=float).flatten()
                for k in sorted(raw_emg.keys())]
    EMGSig = np.column_stack(channels)  # (n_samples, n_channels)
elif isinstance(raw_emg, np.ndarray) and raw_emg.dtype == object:
    # Object array of 1-D channel vectors
    channels = [np.asarray(raw_emg.flat[k], dtype=float).flatten()
                for k in range(raw_emg.size)]
    EMGSig = np.column_stack(channels)
else:
    EMGSig = np.asarray(raw_emg, dtype=float)
    # h5py transposes 2-D arrays vs MATLAB — fix if needed
    # Expected: (n_samples, 16). If shape is (16, n_samples), transpose.
    if EMGSig.ndim == 2 and EMGSig.shape[0] == 16 and EMGSig.shape[1] != 16:
        EMGSig = EMGSig.T

print(f"  EMGSig shape: {EMGSig.shape}  (samples × channels)")

numMUs = len(MUPulses)
numSamples = len(force_signal)
timeVec = np.arange(numSamples) / fsamp

print(f"  Motor units : {numMUs}")
print(f"  Samples     : {numSamples}")
print(f"  Fs          : {fsamp} Hz")

# Task 1.1 – MU Spike Trains and Force Signal

print("\nTask 1.1: MU spike raster + force...")

# Build binary firing matrix  (numMUs × numSamples)
firingMatrix = np.zeros((numMUs, numSamples), dtype=bool)
for i, pulses in enumerate(MUPulses):
    idx = np.asarray(pulses, dtype=int) - 1  # 1-based → 0-based
    idx = idx[(idx >= 0) & (idx < numSamples)]
    firingMatrix[i, idx] = True

fig1, ax1 = plt.subplots(figsize=(12, 6))
fig1.canvas.manager.set_window_title('Task 1.1: MU Spike Trains and Force Signal')

ax1_right = ax1.twinx()

# Raster plot
for i in range(numMUs):
    spike_times = np.where(firingMatrix[i])[0] / fsamp
    ax1.vlines(spike_times, i + 0.55, i + 1.45, color='black', linewidth=0.8)

ax1.set_ylabel('Motor Unit Number')
ax1.set_ylim(0.5, numMUs + 0.5)
ax1.set_yticks(range(1, numMUs + 1))

ax1_right.plot(timeVec, force_signal, color='tab:blue', linewidth=1.5, label='Force')
ax1_right.set_ylabel('Force (N)', color='tab:blue')
ax1_right.set_ylim(0, np.max(force_signal) * 1.1)
ax1_right.tick_params(axis='y', labelcolor='tab:blue')

ax1.set_xlabel('Time (s)')
ax1.set_title('Task 1.1: MU Spike Trains and Force Signal')
ax1.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Task 1.2 – Sorted MU Spike Trains and Force Signal

print("Task 1.2: Sorted spike raster + force...")

firstFiringSamples = np.array([
    np.min(np.asarray(p, dtype=int)) for p in MUPulses
])
sortIdx = np.argsort(firstFiringSamples)

MUPulses_sorted = [MUPulses[i] for i in sortIdx]
firingMatrix_sorted = firingMatrix[sortIdx, :]

fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.canvas.manager.set_window_title('Task 1.2: Sorted MU Spike Trains and Force Signal')

ax2_right = ax2.twinx()

for i in range(numMUs):
    spike_times = np.where(firingMatrix_sorted[i])[0] / fsamp
    ax2.vlines(spike_times, i + 0.55, i + 1.45, color='black', linewidth=0.8)

ax2.set_ylabel('Motor Unit Number (Sorted)')
ax2.set_ylim(0.5, numMUs + 0.5)
ax2.set_yticks(range(1, numMUs + 1))

ax2_right.plot(timeVec, force_signal, color='tab:blue', linewidth=1.5, label='Force')
ax2_right.set_ylabel('Force (N)', color='tab:blue')
ax2_right.set_ylim(0, np.max(force_signal) * 1.1)
ax2_right.tick_params(axis='y', labelcolor='tab:blue')

ax2.set_xlabel('Time (s)')
ax2.set_title('Task 1.2: Sorted MU Spike Trains and Force Signal')
ax2.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Task 1.3 – Instantaneous Discharge Rate (IDR)

print("Task 1.3: Instantaneous discharge rate...")

mu_indices = [0, 1]  # 0-based indices into MUPulses_sorted
colors_idr = ['blue', 'red']

fig3, ax3 = plt.subplots(figsize=(12, 5))
fig3.canvas.manager.set_window_title('Task 1.3: Instantaneous Discharge Rate')

ax3_right = ax3.twinx()

for color, idx in zip(colors_idr, mu_indices):
    spikes_samples = np.asarray(MUPulses_sorted[idx], dtype=int) - 1
    spikes_time = spikes_samples / fsamp
    isi = np.diff(spikes_time)
    idr = 1.0 / isi
    idr_time = spikes_time[:-1]
    ax3.scatter(idr_time, idr, s=20, color=color,
                label=f'MU #{idx + 1}', zorder=3)

ax3.set_ylabel('Discharge Rate (pps)')
ax3.set_ylim(0, 45)
ax3.set_xlabel('Time (s)')

ax3_right.plot(timeVec, force_signal, 'k-', linewidth=1.5, label='Force')
ax3_right.set_ylabel('Force (N)')
ax3_right.set_ylim(0, np.max(force_signal) * 1.1)

# Combine legends from both axes
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_right.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

ax3.set_title('Task 1.3: Instantaneous Discharge Rate of Two MUs and Force Signal')
ax3.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Task 2.1 – Spike-Triggered Averaging (STA) and MUAP Shapes

print("Task 2.1: Spike-triggered averaging...")

STA_window_sec = 0.05
STA_mean = spike_triggered_averaging(EMGSig, MUPulses_sorted, STA_window_sec, fsamp)

chosenMU = 0  # 0-based (MU #1)
muap_grid = STA_mean[chosenMU]

# Flatten 8×2 to list of 16 waveforms (E1–E16)
ordered_muaps = []
for col in range(2):
    for row in range(8):
        ordered_muaps.append(muap_grid[row][col])

# Y-limits
all_vals = np.concatenate([w for w in ordered_muaps if w is not None])
y_limit = (all_vals.min(), all_vals.max()) if len(all_vals) else (-0.1, 0.1)
if y_limit[0] == y_limit[1]:
    y_limit = (-0.1, 0.1)

half_win_samples = int(STA_window_sec * fsamp / 2)
t_sta = np.linspace(-STA_window_sec / 2 * 1000,
                    STA_window_sec / 2 * 1000,
                    2 * half_win_samples + 1)

fig4, axes4 = plt.subplots(16, 1, figsize=(8, 14),
                           gridspec_kw={'hspace': 0.1})
fig4.canvas.manager.set_window_title('Task 2.1: MUAP Shapes 16x1')

for i in range(16):
    # E1 at bottom → tile in reverse order
    ax = axes4[15 - i]
    waveform = ordered_muaps[i]
    if waveform is not None:
        ax.plot(t_sta, waveform, 'b', linewidth=1)
    ax.set_ylim(y_limit)
    ax.set_ylabel(f'E{i + 1}', rotation=0, ha='right', va='center', fontsize=7)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.yaxis.set_tick_params(labelsize=6)
    if i == 0:
        ax.set_xlabel('Time (ms)')
    else:
        ax.set_xticklabels([])

fig4.suptitle(f'Task 2.1: MUAP Shapes of 16 Channels for MU #{chosenMU + 1}')
plt.tight_layout()

# Task 3.1 – Maximum Peak-to-Peak Amplitude

print("Task 3.1: Max peak-to-peak amplitude...")

max_P2P = np.zeros(numMUs)

for n in range(numMUs):
    muap_g = STA_mean[n]
    p2p_values = []
    for col in range(2):
        for row in range(8):
            waveform = muap_g[row][col]
            if waveform is not None:
                p2p_values.append(waveform.max() - waveform.min())
            else:
                p2p_values.append(0.0)
    max_P2P[n] = max(p2p_values)

fig5, ax5 = plt.subplots(figsize=(10, 5))
fig5.canvas.manager.set_window_title('Task 3.1: Max MUAP Amplitude')

mu_nums = np.arange(1, numMUs + 1)
ax5.stem(mu_nums, max_P2P, linefmt='C0-', markerfmt='C0o', basefmt='k-')
ax5.set_title('Task 3.1: Maximum Peak-to-Peak Amplitude per Motor Unit')
ax5.set_xlabel('Motor Unit Number (Sorted by Recruitment)')
ax5.set_ylabel('Max P2P Amplitude (mV/μV)')
ax5.set_xticks(mu_nums)
ax5.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Task 3.2 – Correlation: Amplitude vs Recruitment Order

print("Task 3.2: Amplitude vs recruitment order correlation...")

recruitmentOrder = np.arange(1, numMUs + 1, dtype=float)
r_value, p_value = pearsonr(recruitmentOrder, max_P2P)
print(f"  Pearson r = {r_value:.4f}  (p = {p_value:.4e})")

p_fit = np.polyfit(recruitmentOrder, max_P2P, 1)
y_fit = np.polyval(p_fit, recruitmentOrder)

fig6, ax6 = plt.subplots(figsize=(8, 5))
fig6.canvas.manager.set_window_title('Task 3.2: Amplitude vs Recruitment Order')

ax6.scatter(recruitmentOrder, max_P2P, s=60, color='teal',
            label='MU Data Points', zorder=3)
ax6.plot(recruitmentOrder, y_fit, 'r--', linewidth=2, label='Linear Trendline')

ax6.set_title(f'Task 3.2: MUAP Amplitude vs Recruitment  (r = {r_value:.3f})')
ax6.set_xlabel('Recruitment Order (First to Last)')
ax6.set_ylabel('Max Peak-to-Peak Amplitude (mV)')
ax6.legend(loc='upper left')
ax6.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Task 4.1 – Recruitment and De-recruitment Force Thresholds

print("Task 4.1: Force thresholds...")

recruitmentThresholds = np.zeros(numMUs)
derecruitmentThresholds = np.zeros(numMUs)

for i, pulses in enumerate(MUPulses_sorted):
    spikes = np.asarray(pulses, dtype=int) - 1  # 0-based
    spikes = spikes[(spikes >= 0) & (spikes < numSamples)]
    recruitmentThresholds[i] = force_signal[spikes[0]]
    derecruitmentThresholds[i] = force_signal[spikes[-1]]

fig7, ax7 = plt.subplots(figsize=(10, 5))
fig7.canvas.manager.set_window_title('Task 4.1: Force Thresholds')

ax7.plot(mu_nums, recruitmentThresholds, 'bo-', linewidth=1.5,
         markerfacecolor='blue', label='Recruitment')
ax7.plot(mu_nums, derecruitmentThresholds, 'ro-', linewidth=1.5,
         markerfacecolor='red', label='De-recruitment')

ax7.set_title('Task 4.1: MU Recruitment and De-recruitment Force Thresholds')
ax7.set_xlabel('Motor Unit (Sorted by Recruitment Order)')
ax7.set_ylabel('Force (N)')
ax7.set_xticks(mu_nums)
ax7.legend(loc='upper left')
ax7.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

# Task 4.2 – MUAP RMS Heatmap (Spatial MU Location)

print("Task 4.2: RMS heatmap...")

rms_matrix = np.zeros((16, numMUs))

for n in range(numMUs):
    muap_g = STA_mean[n]
    idx = 0
    for col in range(2):
        for row in range(8):
            waveform = muap_g[row][col]
            if waveform is not None:
                rms_matrix[idx, n] = np.sqrt(np.mean(waveform ** 2))
            idx += 1

fig8, ax8 = plt.subplots(figsize=(12, 7))
fig8.canvas.manager.set_window_title('Task 4.2: MU Location Heatmap')

im = ax8.imshow(rms_matrix, aspect='auto', cmap='jet',
                origin='lower',
                extent=[0.5, numMUs + 0.5, 0.5, 16.5])

cbar = fig8.colorbar(im, ax=ax8)
cbar.set_label('RMS Amplitude (mV)')

ax8.set_title('Task 4.2: Heatmap of MUAP RMS Values (Spatial Location)')
ax8.set_xlabel('Motor Unit Number (Sorted)')
ax8.set_ylabel('Channel Number (Electrode E1 to E16)')
ax8.set_xticks(np.arange(1, numMUs + 1))
ax8.set_yticks(np.arange(1, 17))
ax8.grid(False)
plt.tight_layout()

# Show all figures

print("\nAll tasks complete. Displaying figures...")
plt.show()
