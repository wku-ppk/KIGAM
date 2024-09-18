get_ipython().run_line_magic('reset', '-sf')
import itasca as it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.font_manager import FontProperties
from scipy.integrate import cumtrapz
from scipy.signal import lsim, TransferFunction
from scipy.interpolate import interp1d

it.command("""
model new
model restore 'SL_Sweep008_result.sav'
""")

plt.close('all')
it.command("python-reset-state false") #the model new command will not affect python environment

print('start ==')

################################
### Plot histories
################################
def plotHistory(subPlotM, subPlotN, xLimit, xValues, yValues, legends, xTitle, yTitle, lineColors, opacities):
    ax1 = plt.subplot(gs[subPlotM, subPlotN])                                                      # This subplot takes first column of the first row

    overall_min_y = float('inf')
    overall_max_y = float('-inf')

    for xValue, yValue, legend, lineColor, opacity in zip(xValues, yValues, legends, lineColors, opacities):
        ax1.plot(xValue, yValue, label=legend, alpha=opacity, color=lineColor)

        overall_min_y = min(overall_min_y, np.min(yValue))
        overall_max_y = max(overall_max_y, np.max(yValue))

    ax1.set_xlim(0.0, xLimit)
    if xTitle == 'Y':
        ax1.set_xlabel('Time (s)', fontsize=20, fontweight='bold', family='Cambria', color='black', labelpad=20)
    if yTitle != 'N':
        ax1.set_ylabel(yTitle, fontsize=20, fontweight='bold', family='Cambria', color='black', labelpad=20)
    ax1.grid(False)
    ax1.legend(frameon=False, prop={'family': 'Verdana', 'size': 8, 'weight': 'bold'}, loc='upper right', shadow=True)
    ax1.tick_params(axis='both', direction='inout', length=10)

    ax1.set_ylim(ax1.get_yticks()[0], ax1.get_yticks()[-1]) # Update the Y-axis limits to coincide with major ticks
    
###############################################################
# Calculate Response Spectrum using Frequency Response Spectrum
###############################################################
def calculateResponseSpectra(matrix, periods, damping_ratio=0.05):
    time = matrix[:, 0].astype(np.float64)     # Extract time vector and ensure it's a 1D array of type float64
    time = time.flatten()
    
    if not np.all(np.diff(time) > 0):     # Ensure time is strictly increasing
        raise ValueError("Time vector must be strictly increasing.")
    
    time_diffs = np.diff(time)     # Check if time steps are uniform
    dt = time_diffs[0]
    if np.allclose(time_diffs, dt):
        uniform_time = time # Time steps are uniform; use the original time vector
        print("Time steps are uniform; using the original time vector.")
    else:
        dt = np.mean(time_diffs) # Handle non-uniform time steps by interpolating onto a uniform grid
        uniform_time = np.arange(time[0], time[-1], dt)
        print("Time steps are not uniform; interpolating onto a uniform grid.")
    
    omega_n = 2 * np.pi / periods  # Natural circular frequencies
    
    responseSpectra = np.zeros((len(periods), matrix.shape[1]))
    responseSpectra[:, 0] = periods
    
    for idx in range(1, matrix.shape[1]):
        signalAcc = matrix[:, idx].astype(np.float64)
        
        if not np.array_equal(uniform_time, time):
            interp_func = interp1d(time, signalAcc, kind='linear', fill_value="extrapolate")
            signalAcc_uniform = interp_func(uniform_time)
        else:
            signalAcc_uniform = signalAcc
        
        Sa = np.zeros_like(periods)
        
        for i, omega in enumerate(omega_n):
            wn = omega  # Natural frequency
            num = [1]  # Numerator
            den = [1, 2 * damping_ratio * wn, wn ** 2]  # Denominator
            system = TransferFunction(num, den)
                
            _, response, _ = lsim(system, U=signalAcc_uniform, T=uniform_time)
                
            Sd = np.max(np.abs(response))         # Spectral Displacement unit follows input
            Sa[i] = Sd * wn ** 2                 # Spectral Acceleration unit follows input
        
        responseSpectra[:, idx] = Sa
    
    return responseSpectra

periods = np.array([
    0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
    0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
    0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
    0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34,
    0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6,
    0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86,
    0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
    1.8, 1.9, 2.0, 3.0, 4.0, 5.0])


################################
### Read FLAC History Array
################################
## Construct Disp, Vel, Acc Matrix of Soil Layers
AccArray = np.array(it.history.get('11', '10')) # Bedrock Acc x

def constArray(AccArray, historyName):
    tempCol = np.array(it.history.get(historyName))
    new_col = tempCol[:, 1].reshape(-1, 1)
    AccArray = np.hstack((AccArray, new_col))
    return AccArray

AccArray = constArray(AccArray, '12') # Bedrock Acc y
AccArray = constArray(AccArray, '13') # Bedrock Acc x
AccArray = constArray(AccArray, '14') # Bedrock !!! Warning This is Velocity -x
AccArray = constArray(AccArray, '15') # Bedrock !!! Warning This is Displacement -x

AccArray = constArray(AccArray, '101')
AccArray = constArray(AccArray, '102')
AccArray = constArray(AccArray, '103')
AccArray = constArray(AccArray, '104')
AccArray = constArray(AccArray, '105')
AccArray = constArray(AccArray, '106')

AccArray = constArray(AccArray, '201')
AccArray = constArray(AccArray, '202')
AccArray = constArray(AccArray, '203')
AccArray = constArray(AccArray, '204')
AccArray = constArray(AccArray, '205')
AccArray = constArray(AccArray, '206')

AccArray = constArray(AccArray, '301')
AccArray = constArray(AccArray, '302')
AccArray = constArray(AccArray, '303')
AccArray = constArray(AccArray, '304')
AccArray = constArray(AccArray, '305')
AccArray = constArray(AccArray, '306')

AccArray_truncated = AccArray[852:,:] # Just to delete garvage values, I don't know why, so I asked Itasca's software forum about this. Let's wait answer

###

time = AccArray_truncated[:, 0].reshape(-1, 1)  # Extract the time array # n x 1 column vector
acc_matrix = AccArray_truncated[:, 1:]  # Extract acceleration signals as a matrix (n x (m-1)) # n x (m-1) matrix
vel_matrix = cumtrapz(acc_matrix, time, axis=0, initial=0) # Perform cumulative integration using matrix calculations # Each column of acc_matrix is integrated with respect to the time vector
VelocityArray = np.hstack((time, vel_matrix)) # Combine the time column with the velocity matrix into a new array

vel_matrix = VelocityArray[:, 1:]  # Extract acceleration signals as a matrix (n x (m-1)) # n x (m-1) matrix
disp_matrix = cumtrapz(vel_matrix, time, axis=0, initial=0) # Perform cumulative integration using matrix calculations # Each column of acc_matrix is integrated with respect to the time vector
DispArray = np.hstack((time, disp_matrix)) # Combine the time column with the velocity matrix into a new array

AccArray_g = np.hstack((time, acc_matrix/9.81))

################################
### Read Centrifuge test recorded array
################################

# Load Excel file with multiple sheets
file_path = 'Sweep_008.xlsx'
sheets = pd.read_excel(file_path, sheet_name=None)  # Load all sheets into a dictionary

# To read a specific sheet (replace 'Sheet1' with your sheet name or index)
sheet_name = 'AP'
df = pd.read_excel(file_path, sheet_name=sheet_name, usecols="A:P")

# Select the data starting from B10 (i.e., row 10, column B)
# Skip the first 9 rows (B10 is in the 10th row) and use the columns B and onward (for time history and acceleration data)
start_row = 9  # B10 corresponds to row index 9 (0-based index)
data = df.iloc[start_row:, 1:]  # Extract data from row 10 (index 9) and column B onwards

# Convert this to a matrix (NumPy array) if needed
matrix = data.to_numpy()

# Drop rows that contain any NaN values
data_clean = data.dropna()

# Convert to matrix
matrix_clean = data_clean.to_numpy()

# Acc. Histories
#Plot and Array Definition
#Acc : hist #

#A1  : 101   ----  A4  :     : 301
#A7  : 102   ----  A8  : 202
#A13 : 103   ----  A15 :     : 303
#            ----  A17 : 204
#A21 : 105   ----  A22 : 205
#            ----  A25 : 206

# Convert to Excel Array
#C7  : 101   ----  F7  :     : 301
#C6  : 102   ----  F6  : 202
#C5  : 103   ----  F5  :     : 303
#            ----  F4  : 204
#C3  : 105   ----  F3  : 205
#            ----  F2  : 206

# Convert to Python Array, MC: Matrix_Clean, AA : Acc_Array
#MC-7 : AA-6  ----  MC-14 :     : AA-18
#MC-6 : AA-7  ----  MC-13 : AA-13
#MC-5 : AA-8  ----  MC-12 :     : AA-20
#             ----  MC-11 : AA-15
#MC-3 : AA-10 ----  MC-10 : AA-16
#             ----  MC- 9 : AA-17


# Calculate response spectra
spectrasTEST = calculateResponseSpectra(matrix_clean, periods, damping_ratio=0.05)
spectrasFLAC = calculateResponseSpectra(AccArray_g, periods, damping_ratio=0.05)

lable_properties = FontProperties(family='Cambria', style='normal', size=18)

fig = plt.figure(figsize=(30, 12))                                                            # Set the figure size (width, height) in inches
gs = gridspec.GridSpec(6, 4, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1])        # Define the GridSpec with specific height and width ratios
gs.update(wspace=0.2, hspace=0.5)                                                             # Adjust the spacing between subplots # Adjust these values to increase/decrease spacing

plotHistory(0, 0, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,7], AccArray_g[:,6]], ['A1', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(0, 1, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,7], spectrasFLAC[:,6]], ['A1', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(0, 2, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,14], AccArray_g[:,18]], ['A4', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(0, 3, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,14], spectrasFLAC[:,18]], ['A4', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])

plotHistory(1, 0, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,6], AccArray_g[:,7]], ['A7', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 1, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,6], spectrasFLAC[:,7]], ['A7', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 2, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,13], AccArray_g[:,13]], ['A8', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(1, 3, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,13], spectrasFLAC[:,13]], ['A8', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])

plotHistory(2, 0, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,5], AccArray_g[:,8]], ['A13', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 1, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,5], spectrasFLAC[:,8]], ['A13', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 2, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,12], AccArray_g[:,20]], ['A15', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(2, 3, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,12], spectrasFLAC[:,20]], ['A15', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])

plotHistory(3, 2, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,11], AccArray_g[:,15]], ['A17', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(3, 3, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,11], spectrasFLAC[:,15]], ['A17', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])

plotHistory(4, 0, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,3], AccArray_g[:,10]], ['A21', 'FLAC'], 'Time(s)', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 1, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,3], spectrasFLAC[:,10]], ['A21', 'FLAC'], 'Time(s)', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 2, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,10], AccArray_g[:,16]], ['A22', 'FLAC'], 'N', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(4, 3, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,10], spectrasFLAC[:,16]], ['A22', 'FLAC'], 'N', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])

plotHistory(5, 2, 12.0, [matrix_clean[:,0], AccArray_g[:,0]], [matrix_clean[:,9], AccArray_g[:,17]], ['A25', 'FLAC'], 'Time(s)', 'Acc. (g)', ['blue', 'red'], [1.0, 0.6])
plotHistory(5, 3, 3.0, [spectrasTEST[:,0], spectrasFLAC[:,0]], [spectrasTEST[:,9], spectrasFLAC[:,17]], ['A25', 'FLAC'], 'Time(s)', 'Sa. (g)', ['blue', 'red'], [1.0, 0.6])

plt.show()







