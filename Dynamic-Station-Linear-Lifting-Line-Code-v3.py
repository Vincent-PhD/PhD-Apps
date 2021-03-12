import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import shift
from scipy.interpolate import CubicSpline
from scipy.interpolate import barycentric_interpolate
import joblib
import itertools
import pandas as pd
from Airfoil_Parametric_Function_Generator import gen_foil
from xfoil_dat_3 import xfoil_calculations
from xfoil_dat_3 import xfoil_calculations_cl

# For every station, calculate the sectional 2D properties for use in LLT

# Foil for station 1

Foil_Station_1 = xfoil_calculations(B = 1.952,
                                       T = 0.125,
                                       P = 3.281,
                                       E = 2.506,
                                       C = 0.0126,
                                       R = 0,
                                       Foil_name = 'Test_Foil.dat',
                                       Reynolds = '2000000',
                                       Mach = '0',
                                       iter = '1000',
                                       alfa_sequence = '-5 5 .25',
                                       Foil_name_xfoil = 'FOIL',
                                       File_Path = "C:\\Users\\Vincent\\Documents\\GitHub\\Advanced-Lifting-Line-Code-Streamlit\\",
                                       saveFlnmAF = 'Save_Airfoil.txt',
                                       xfoilFlnm = 'xfoil_input.txt',
                                       saveFlnmCd = 'Save_Cd.txt',
                                       saveFlnmCp = 'Save_Cp.txt')

Slope_Station_1 = Foil_Station_1[0]
Zero_Lift_AoA_Station_1 = Foil_Station_1[1]

# Foil for station 2

Foil_Station_2 = xfoil_calculations(B = 1.952,
                                       T = 0.125,
                                       P = 3.281,
                                       E = 2.506,
                                       C = 0.0126,
                                       R = 0,
                                       Foil_name = 'Test_Foil_2.dat',
                                       Reynolds = '2000000',
                                       Mach = '0',
                                       iter = '1000',
                                       alfa_sequence = '-5 5 .25',
                                       Foil_name_xfoil = 'FOIL_2',
                                       File_Path = "C:\\Users\\Vincent\\Documents\\GitHub\\Advanced-Lifting-Line-Code-Streamlit\\",
                                       saveFlnmAF = 'Save_Airfoil.txt',
                                       xfoilFlnm = 'xfoil_input.txt',
                                       saveFlnmCd = 'Save_Cd.txt',
                                       saveFlnmCp = 'Save_Cp.txt')

Slope_Station_2 = Foil_Station_2[0]
Zero_Lift_AoA_Station_2 = Foil_Station_2[1]

# Foil for station 3

Foil_Station_3 = xfoil_calculations(B = 1.952,
                                       T = 0.125,
                                       P = 3.281,
                                       E = 2.506,
                                       C = 0.0126,
                                       R = 0,
                                       Foil_name = 'Test_Foil_3.dat',
                                       Reynolds = '2000000',
                                       Mach = '0',
                                       iter = '1000',
                                       alfa_sequence = '-5 5 .25',
                                       Foil_name_xfoil = 'FOIL_3',
                                       File_Path = "C:\\Users\\Vincent\\Documents\\GitHub\\Advanced-Lifting-Line-Code-Streamlit\\",
                                       saveFlnmAF = 'Save_Airfoil.txt',
                                       xfoilFlnm = 'xfoil_input.txt',
                                       saveFlnmCd = 'Save_Cd.txt',
                                       saveFlnmCp = 'Save_Cp.txt')

Slope_Station_3 = Foil_Station_3[0]
Zero_Lift_AoA_Station_3 = Foil_Station_3[1]

# Foil for station 4

Foil_Station_4 = xfoil_calculations(B = 1.952,
                                       T = 0.125,
                                       P = 3.281,
                                       E = 2.506,
                                       C = 0.0126,
                                       R = 0,
                                       Foil_name = 'Test_Foil_4.dat',
                                       Reynolds = '2000000',
                                       Mach = '0',
                                       iter = '1000',
                                       alfa_sequence = '-5 5 .25',
                                       Foil_name_xfoil = 'FOIL_4',
                                       File_Path = "C:\\Users\\Vincent\\Documents\\GitHub\\Advanced-Lifting-Line-Code-Streamlit\\",
                                       saveFlnmAF = 'Save_Airfoil.txt',
                                       xfoilFlnm = 'xfoil_input.txt',
                                       saveFlnmCd = 'Save_Cd.txt',
                                       saveFlnmCp = 'Save_Cp.txt')

Slope_Station_4 = Foil_Station_4[0]
Zero_Lift_AoA_Station_4 = Foil_Station_4[1]

# Foil for station 4

Foil_Station_5 = xfoil_calculations(B = 1.952,
                                       T = 0.125,
                                       P = 3.281,
                                       E = 2.506,
                                       C = 0.0126,
                                       R = 0,
                                       Foil_name = 'Test_Foil_5.dat',
                                       Reynolds = '2000000',
                                       Mach = '0',
                                       iter = '1000',
                                       alfa_sequence = '-5 5 .25',
                                       Foil_name_xfoil = 'FOIL_5',
                                       File_Path = "C:\\Users\\Vincent\\Documents\\GitHub\\Advanced-Lifting-Line-Code-Streamlit\\",
                                       saveFlnmAF = 'Save_Airfoil.txt',
                                       xfoilFlnm = 'xfoil_input.txt',
                                       saveFlnmCd = 'Save_Cd.txt',
                                       saveFlnmCp = 'Save_Cp.txt')

Slope_Station_5 = Foil_Station_5[0]
Zero_Lift_AoA_Station_5 = Foil_Station_5[1]


# Create a function that determines the slope of a lift curve (linear)

def determine_slope(lift_array, aoa_array):
    slope = (np.max(lift_array)-np.min(lift_array)) / ((aoa_array[np.where(lift_array == np.max(lift_array) )]*(np.pi/180)) - (aoa_array[np.where(lift_array == np.min(lift_array) )]*np.pi/180) )
    return slope

    # Load the Deep-Learning/ Machine Learning Model


def load_ml_models(Profile_Drag_Model_Name,Lift_Model_Name, Moment_Model_Name, Transition_Model_Name):
    print('Importing ML Drag Model ... ... ...')
    loaded_drag_model = joblib.load(Profile_Drag_Model_Name)
    print('Importing ML Lift Model ... ... ...')
    loaded_lift_model = joblib.load(Lift_Model_Name)
    print('Importing ML Moment Model ... ... ...')
    loaded_moment_model = joblib.load(Moment_Model_Name)
    print('Importing ML Transition Model ... ... ...')
    loaded_transition_model = joblib.load(Transition_Model_Name)

    return loaded_drag_model, loaded_lift_model, loaded_moment_model, loaded_transition_model, print('ML model uploaded')

# Define Reynolds Function

def get_Reynolds(Chord_Lenght, Free_Stream_Velocity = 65, Density = 1.2253, Dynamic_Viscosity = 0.0000173320):
    Chord_Reynolds_Number = []
    for i in range(0, len(Chord_Lenght)):
        Re_i = (Chord_Lenght[i]*Free_Stream_Velocity*Density)/(Dynamic_Viscosity)
        Chord_Reynolds_Number = np.append(Chord_Reynolds_Number, Re_i)

    return Chord_Reynolds_Number


def get_wing_area(Wing_Roots, Wing_Tips, Station_Lenghts):
    Wing_Area = []

    for i in range(0,len(Wing_Roots)):

        STA_i = (Wing_Roots[i]-Wing_Tips[i]) * (Station_Lenghts[i]/2) + (Wing_Tips[i]*Station_Lenghts[i])
        Wing_Area = np.append(Wing_Area, STA_i)

    return Wing_Area

# Generate Theta Distribution

def get_Theta(N, Theta_Constant):

    Theta = []

    for i in range(0, N):

        Theta_i = Theta_Constant + (i * Theta_Constant)
        Theta = np.append(Theta, Theta_i)

    return Theta

# Generate Alpha Distribution

def get_Alpha(N, alpha_twist, i_w, Alpha_Constant):

    Alpha = alpha_twist + i_w
    for i in range(0, N):
        if i == 0:
            Alpha_i = Alpha + (Alpha_Constant)
        else:
            Alpha_i = (alpha_twist + i_w) + (i * Alpha_Constant)
        Alpha = np.append(Alpha, Alpha_i)

    return Alpha

# Calculate Station Y-Location

def get_station_Y_location(Station_Lenght):

    Station_position = []

    for i in range(0, len(Station_Lenght)):
        if i <= 0:
            Station_Position_i = Station_Lenght[i]
        if i == 1:
            Station_Position_i = (Station_Lenght[i] +
                                  Station_Lenght[i - 1])
        if i == 2:
            Station_Position_i = (Station_Lenght[i] +
                                  Station_Lenght[i - 1] +
                                  Station_Lenght[i - 2])
        if i == 3:
            Station_Position_i = (Station_Lenght[i] +
                                  Station_Lenght[i - 1] +
                                  Station_Lenght[i - 2] +
                                  Station_Lenght[i - 3])
        if i == 4:
            Station_Position_i = (Station_Lenght[i] +
                                  Station_Lenght[i - 1] +
                                  Station_Lenght[i - 2] +
                                  Station_Lenght[i - 3] +
                                  Station_Lenght[i - 4])
        if i == 5:
            Station_Position_i = (Station_Lenght[i] +
                                  Station_Lenght[i - 1] +
                                  Station_Lenght[i - 2] +
                                  Station_Lenght[i - 3] +
                                  Station_Lenght[i - 4] +
                                  Station_Lenght[i - 5])

        Station_position = np.append(Station_position, Station_Position_i)

    return Station_position

# Generate Linearly Spaced Stations

def generate_stations(Theta, N, b):

    Station_generated = []

    for i in range(0, N):
        Gen_i = ((b) / (2)) * np.cos(Theta[i])
        Station_generated = np.append(Station_generated, Gen_i)


    return Station_generated

# Find nearest value function

def find_nearest(array, value):

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()

    return array[idx]

# Find nearest value between generated stations and actual station
def get_nearest_station_lenght(Station_Lenght, Station_generated, Station_position):

    Nearest_Value = []

    for i in range(0,len(Station_Lenght)):

        NV_i = find_nearest(Station_generated, Station_position[i])
        Nearest_Value = np.append(Nearest_Value, NV_i)
    #print('Nearest Value between gen and actual',Nearest_Value)

    return Nearest_Value

# Find index value of Nearest Number Between generated Stations and Actual Stations

def get_index(Station_Lenght, Station_generated, Nearest_Value):

    Index_Value = []

    for i in range(0,len(Station_Lenght)):

        IV_i = np.where(Station_generated == Nearest_Value[i])
        Index_Value = np.append(Index_Value, IV_i)

    #print('Index value', Index_Value)

    return Index_Value

# Create length scalar for Root Chord division

def get_length_scalar(Station_Lenght, Index_Value):

    Length_Scalar = []

    for i in range(0,len(Station_Lenght)):
        if i == 0:
            LS_i = 0
        else:
            LS_i = Index_Value[i] - Index_Value[i-1]

            Length_Scalar = np.abs(np.append(Length_Scalar, LS_i))

    #print('Lenght Scalar', Length_Scalar)

    return Length_Scalar

def get_MAC(Index_Value, Root_Chords, Length_Scalar, N):

    MAC = []
    MAC_i = Root_Chords[0]
    for i in range(0,N):

        if i < Index_Value[0] and i >= Index_Value[1]:
            MAC_i = MAC_i - ((Root_Chords[0] - Root_Chords[1]) / (Length_Scalar[0]))
            MAC = np.append(MAC, MAC_i)

    MAC_i = Root_Chords[1]
    for i in range(0, N):

        if i < Index_Value[1] and i >= Index_Value[2]:
            MAC_i = MAC_i - ((Root_Chords[1] - Root_Chords[2]) / (Length_Scalar[1]))
            MAC = np.append(MAC, MAC_i)

    MAC_i = Root_Chords[2]
    for i in range(0, N):

        if i < Index_Value[2] and i >= Index_Value[3]:
            MAC_i = MAC_i - ((Root_Chords[2] - Root_Chords[3]) / (Length_Scalar[2]))
            MAC = np.append(MAC, MAC_i)

    MAC_i = Root_Chords[3]
    for i in range(0, N):

        if i < Index_Value[3] and i >= Index_Value[4]:
            MAC_i = MAC_i - ((Root_Chords[3] - Root_Chords[4]) / (Length_Scalar[3]))
            MAC = np.append(MAC, MAC_i)

    MAC_i = Root_Chords[4]
    for i in range(0, N):

        if i < Index_Value[4] and i >= Index_Value[5]:
            MAC_i = MAC_i - ((Root_Chords[4] - Root_Chords[5]) / (Length_Scalar[4]))
            MAC = np.append(MAC, MAC_i)

    MAC = np.insert(MAC, 0, Root_Chords[0])
    MAC = np.flip(MAC)

    return MAC
# Generate Station lift curve slope and zero lift angle of attack information

def station_2d_properties(N, Station_Alpha_2d, Station_lift_curve_slope, Number_of_stations):

    Station_wise_lift_curve_slope = []
    Station_wise_zero_lift_aoa = []

    for i in range(0,N):
        if i <= ((N/Number_of_stations)*1):
            Station_wise_zero_lift_aoa_i = Station_Alpha_2d[4]
            Station_wise_lift_curve_slope_i = Station_lift_curve_slope[4]

        elif i <= ((N/Number_of_stations)*2):
            Station_wise_zero_lift_aoa_i = Station_Alpha_2d[3]
            Station_wise_lift_curve_slope_i = Station_lift_curve_slope[3]
        elif i <= ((N/Number_of_stations)*3):
            Station_wise_zero_lift_aoa_i = Station_Alpha_2d[2]
            Station_wise_lift_curve_slope_i = Station_lift_curve_slope[2]
        elif i <= ((N/Number_of_stations)*4):
            Station_wise_zero_lift_aoa_i = Station_Alpha_2d[1]
            Station_wise_lift_curve_slope_i = Station_lift_curve_slope[1]
        else:
            Station_wise_zero_lift_aoa_i = Station_Alpha_2d[0]
            Station_wise_lift_curve_slope_i = Station_lift_curve_slope[0]


        Station_wise_zero_lift_aoa = np.append(Station_wise_zero_lift_aoa , Station_wise_zero_lift_aoa_i)
        Station_wise_lift_curve_slope = np.append(Station_wise_lift_curve_slope, Station_wise_lift_curve_slope_i)

    return Station_wise_zero_lift_aoa, Station_wise_lift_curve_slope

# Calculate mu matrix

def get_mu(N, Station_wise_lift_curve_slope, MAC, b):

    mu = []

    for i in range(0,N):

        mu_i = MAC[i] * (Station_wise_lift_curve_slope[i]) / (4 * b)
        mu = np.append(mu, mu_i)

    return mu

# Determine Matrix LHS

def get_lhs(mu, Alpha, Station_wise_zero_lift_aoa, N):

    LHS = []

    for i in range(0,N):

        LHS_i = mu[i]*((Alpha[i]-Station_wise_zero_lift_aoa[i]))/(180/np.pi)
        LHS = np.append(LHS, LHS_i)

    return LHS

# Determine Matrix RHS

def get_rhs(mu, Theta, N):

    RHS = []

    for i in range(0,N):

        for j in range(0, N):

            RHS_i = np.sin((2*(j+1)-1)*Theta[i])*(1+(mu[i]*(2*(j+1)-1))/np.sin(Theta[i]))
            RHS = np.append(RHS, RHS_i)

            # Reshape Data

    RHS = np.reshape(RHS, (N, N))

    return RHS

# Solve for A coefficients

def get_A_coeff(RHS, LHS):

    A_coeff = np.linalg.solve(RHS, LHS)

    return A_coeff

# Calculate Wing Gamma

def get_wing_gamma(N, A_coeff):

    gamma = []

    for i in range(2, N):

        gamma_i = (2 * i - 1) * (((A_coeff[i - 1]) / (A_coeff[0]))) ** 2
        gamma = np.append(gamma, gamma_i)
        gamma = np.sum(gamma)

    return gamma

# Calculate wing performance and efficiency

def get_wing_performance(gamma, AR, A_coeff, Number_of_stations):

    Wing_efficiency = 1 / (1 + gamma)
    Wing_Lift = np.pi * AR * A_coeff[0]
    Wing_Induced_Drag = (Wing_Lift ** 2) / (np.pi * AR * Wing_efficiency)

    print(f'Total wing stations: {Number_of_stations}')
    print(f'Wing efficiency: {Wing_efficiency * 100} %')
    print(f'Wing dimensionless lift: {Wing_Lift}')
    print(f'Wing dimensionless induced drag: {Wing_Induced_Drag}')

    return Wing_efficiency, Wing_Lift, Wing_Induced_Drag

# Calculate span wise lift distribution

def get_wing_local_lift(N, A_coeff, Theta):

    Local_Wing_Lift = []

    for i in range(0, N):

        for j in range(0, N):

            Local_Wing_Lift_i = A_coeff[j] * np.sin((2 * (j + 1) - 1) * Theta[i])

            Local_Wing_Lift = np.append(Local_Wing_Lift, Local_Wing_Lift_i)

    # Reshape Local Wing Lift Matrix

    Local_Wing_Lift = Local_Wing_Lift.reshape(N, N)


    return Local_Wing_Lift



# Create summation matrix for span wise lift distribution calculation

def get_local_gamma_sum(N, Local_Wing_Lift, b, MAC):

    local_Gamma_Sum = []

    for i in range(0, N):

        Local_Gamma_Sum_i = 4*(np.sum(Local_Wing_Lift[i]))*((b)/(MAC[i]))

        print(f'Wing local lift for station {np.abs(i-N)} is {Local_Gamma_Sum_i}')

        local_Gamma_Sum = np.append(local_Gamma_Sum, Local_Gamma_Sum_i)

    return local_Gamma_Sum



# Plot Elliptical Distribution (Ideal)

def plot_ideal_LD(Station_generated, local_Gamma_Sum, b):

    u = 0  # x-position of the center
    v = 0  # y-position of the center
    a = b / 2  # radius on the x-axis i.e half span
    z = local_Gamma_Sum[-1]  # radius on the y-axis i.e max lift at station

    t = np.linspace(0, np.pi / 2, 100)
    plt.plot(u + a * np.cos(t), v + z * np.sin(t), label='Elliptical Distribution')
    plt.grid(color='lightgray', linestyle='--')
    plt.xlabel('Wing Span [m]')
    plt.ylabel('Local Lift Coefficient')
    plt.title('Ideal vs Actual Station-Wise Lift Distribution')
    plt.legend()

    # Plot Station wise lift distribution

    plt.plot(Station_generated, local_Gamma_Sum, label='Actual Lift Distribution', marker=',')
    plt.legend()
    #plt.show()

    return



# Wing Plot

def plot_wing_LD(Station_generated, N, C_root, MAC, local_Gamma_Sum):

    plt.plot(Station_generated, np.zeros(N)*-1, label = 'Wing', marker = ',', c = 'darkblue')
    plt.plot(np.zeros(N), np.linspace(0,C_root,num=N)*-1, marker = ',', c = 'darkblue')
    plt.plot(Station_generated, MAC*-1, marker = ',', c = 'darkblue')
    plt.plot( (np.zeros(N)+Station_generated[0]),np.linspace(0,MAC[0],num=N)*-1,  marker = ',', c = 'darkblue')
    plt.plot(Station_generated, (local_Gamma_Sum*np.average(Station_generated)), label = 'Actual Lift Distribution', marker = ',',c = 'darkred')
    plt.plot(Station_generated*-1, np.zeros(N)*-1, marker = ',', c = 'darkblue')
    plt.plot(np.zeros(N)*-1, np.linspace(0,C_root,num=N)*-1, marker = ',', c = 'darkblue')
    plt.plot(Station_generated*-1, MAC*-1, marker = ',', c = 'darkblue')
    plt.plot( (np.zeros(N)+Station_generated[0])*-1,np.linspace(0,MAC[0],num=N)*-1,  marker = ',', c = 'darkblue')
    plt.plot(Station_generated*-1, (local_Gamma_Sum*np.average(Station_generated)), marker = ',',c = 'darkred')
    plt.grid(color='lightgray',linestyle='--')
    plt.xlabel('Wing Span [m]')
    plt.ylabel('Lift Distribution')
    plt.title('Wing Lift Distribution')
    plt.legend()
    #plt.show()

    return

def Dynamic_LLT_Simulation(N = 50,
                           AR = 22.6,
                           alpha_twist = 0.0,
                           i_w = 2.0 ,
                           Station_Alpha_2d = np.array([-3.85, -3.85, -3.85, -3.85, -3.85]),
                           Station_lift_curve_slope = np.array([Slope_Station_1, Slope_Station_2, Slope_Station_3, Slope_Station_4, Slope_Station_5]),
                           Station_Lenght = np.array([0, 2.200, 1.980, 1.800, 0.890, 0.670]),
                           Root_Chords = np.array([0.820, 0.800, 0.700, 0.535, 0.419, 0.210])):

    # Initial Calculations

    b = np.sum(Station_Lenght) * 2  # wing span [m]
    C_root = Root_Chords[0]  # Root Chord
    Theta_Constant = np.pi / (2 * N)  # Theta partitioning
    Alpha_Constant = np.abs(alpha_twist) / (N - 1)  # Alpha partitioning
    Number_of_stations = len(Station_Alpha_2d)  # Number of wing stations

    Theta = get_Theta(N,Theta_Constant)
    Alpha = get_Alpha(N, alpha_twist, i_w, Alpha_Constant)
    Station_position = get_station_Y_location(Station_Lenght)
    Station_generated = generate_stations(Theta, N, b)
    Nearest_Value = get_nearest_station_lenght(Station_Lenght, Station_generated, Station_position)
    Index_Value = get_index(Station_Lenght, Station_generated, Nearest_Value)
    Length_Scalar = get_length_scalar(Station_Lenght, Index_Value)
    MAC = get_MAC(Index_Value, Root_Chords, Length_Scalar, N)
    Station_wise_zero_lift_aoa = station_2d_properties(N, Station_Alpha_2d, Station_lift_curve_slope, Number_of_stations)[0]
    Station_wise_lift_curve_slope = station_2d_properties(N, Station_Alpha_2d, Station_lift_curve_slope, Number_of_stations)[1]
    mu = get_mu(N, Station_wise_lift_curve_slope, MAC, b)
    LHS = get_lhs(mu, Alpha, Station_wise_zero_lift_aoa, N)
    RHS = get_rhs(mu, Theta, N)
    A_coeff = get_A_coeff(RHS, LHS)
    gamma = get_wing_gamma(N, A_coeff)
    Wing_efficiency = get_wing_performance(gamma, AR, A_coeff, Number_of_stations)[0]
    Wing_Lift = get_wing_performance(gamma, AR, A_coeff, Number_of_stations)[1]
    Wing_Induced_Drag = get_wing_performance(gamma, AR, A_coeff, Number_of_stations)[2]
    Local_Wing_Lift = get_wing_local_lift(N, A_coeff, Theta)
    local_Gamma_Sum = get_local_gamma_sum(N, Local_Wing_Lift, b, MAC)
    Chord_Reynolds = get_Reynolds(Station_Lenght, Free_Stream_Velocity = 65, Density = 1.2253, Dynamic_Viscosity = 0.0000173320)
    Station_Areas = get_wing_area(Root_Chords, shift(Root_Chords,1, cval=0), Station_Lenght)
    Station_Areas = Station_Areas[1:]

    # Determine average lift for every station

    Average_Lift_Satation_1 = np.average(local_Gamma_Sum[0:9]);   print(f'Local Wing Lift Station 1 is: {Average_Lift_Satation_1}')
    Average_Lift_Satation_2 = np.average(local_Gamma_Sum[9:19]);  print(f'Local Wing Lift Station 2 is: {Average_Lift_Satation_2}')
    Average_Lift_Satation_3 = np.average(local_Gamma_Sum[19:29]); print(f'Local Wing Lift Station 3 is: {Average_Lift_Satation_3}')
    Average_Lift_Satation_4 = np.average(local_Gamma_Sum[29:39]); print(f'Local Wing Lift Station 4 is: {Average_Lift_Satation_4}')
    Average_Lift_Satation_5 = np.average(local_Gamma_Sum[39:49]); print(f'Local Wing Lift Station 5 is: {Average_Lift_Satation_5}')

    # Determine profile drag for every station
    # NNNNNNNNNNNNNNNNNBBBBBBBBBBBBBBBBBBBBBBBBB The Reynolds number still needs to change dynamically

    Profile_Drag_Station_1_Calc = xfoil_calculations_cl(B=1.952,
                                                        T=0.125,
                                                        P=3.281,
                                                        E=2.506,
                                                        C=0.0126,
                                                        R=0,
                                                        Foil_name='Test_Foil.dat',
                                                        Reynolds='2000000',
                                                        Mach='0',
                                                        iter='1000',
                                                        Cl= f'{Average_Lift_Satation_1}',
                                                        Foil_name_xfoil='FOIL',
                                                        File_Path= "C:\\Users\\Vincent\\Documents\\GitHub\\Advanced-Lifting-Line-Code-Streamlit\\",
                                                        saveFlnmAF='Save_Airfoil.txt',
                                                        xfoilFlnm='xfoil_input.txt',
                                                        saveFlnmCd='Save_Cd.txt',
                                                        saveFlnmCp='Save_Cp.txt')

    Profile_Drag_Station_1 = Profile_Drag_Station_1_Calc[1]

    print(f'Local Profile Drag Station 1 is: {Profile_Drag_Station_1}')


    Average_Station_Lift = [Average_Lift_Satation_1, Average_Lift_Satation_2, Average_Lift_Satation_3, Average_Lift_Satation_4, Average_Lift_Satation_5]
    Station_Profile_Drag = [Profile_Drag_Station_1]

    # Plot Results

    plot_ideal_LD(Station_generated, local_Gamma_Sum, b)
    plot_wing_LD(Station_generated, N, C_root, MAC, local_Gamma_Sum)
    plt.show()

    print('Wing Station Area Matrix',Station_Areas)


    return Wing_efficiency, Wing_Lift, Wing_Induced_Drag, Chord_Reynolds, Station_Areas, Local_Wing_Lift, Average_Station_Lift, Station_Profile_Drag

# Define a function that runs the LLT function for multiple AoA's

def LLT_simulation(AoA_Range = np.array([-3, -2, -1, -0.5,0 ,0.5 , 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17])):
    Wing_Induced_Drag_Matrix = []
    Wing_Lift_Matrix = []
    Wing_efficiency_Matrix = []
    for AoA in AoA_Range:
        LLT_i = Dynamic_LLT_Simulation(i_w = AoA, alpha_twist= -1.5)

        Wing_efficiency_Matrix = np.append(Wing_efficiency_Matrix, LLT_i[0])
        Wing_Lift_Matrix = np.append(Wing_Lift_Matrix, LLT_i[1])
        Wing_Induced_Drag_Matrix = np.append(Wing_Induced_Drag_Matrix, LLT_i[2])

    plt.plot(AoA_Range, Wing_efficiency_Matrix*100)
    plt.title('Simulation Results: Wing Efficiency vs Angle of Attack')
    plt.grid(True)
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('Wing Efficiency [%]')
    plt.show()

    plt.plot(AoA_Range, Wing_Lift_Matrix)
    plt.title('Simulation Results: Wing Total Lift vs Angle of Attack')
    plt.grid(True)
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('Wing Lift')
    plt.show()

    plt.plot(AoA_Range, Wing_Induced_Drag_Matrix)
    plt.title('Simulation Results: Wing Induced Drag vs Angle of Attack')
    plt.grid(True)
    plt.xlabel('Angle of Attack [deg]')
    plt.ylabel('Wing Induced Drag')
    plt.show()

    plt.plot(Wing_Induced_Drag_Matrix, Wing_Lift_Matrix)
    plt.title('Simulation Results: Wing Induced Drag vs Wing Total Lift')
    plt.grid(True)
    plt.xlabel('Wing Induced Drag')
    plt.ylabel('Wing Lift')
    plt.show()

    return Wing_efficiency_Matrix, Wing_Lift_Matrix, Wing_Induced_Drag_Matrix

# Calculate Fuselage Drag (Fuselage glider speed and glider drag calculated in CFD simulations at a AoA and calculated speed determined from LLT

def fuselage_drag_interpolater(glider_speed_matrix, glider_drag_matrix, calculated_speed_matrix):

  fuselage_glider_drag = []

  for i in range(0, len(calculated_speed_matrix)):

    cs = CubicSpline(glider_speed_matrix , glider_drag_matrix)
    drag_i = cs(calculated_speed_matrix[i], extrapolate=True)
    fuselage_glider_drag = np.append(fuselage_glider_drag, drag_i)
    print(f'For Glider Calculated Speed of {calculated_speed_matrix[i]}, the Glider Drag is {drag_i} Newton')

  return fuselage_glider_drag

# Calculate the glider flight speed from LLT Wing Lift Result

def flight_speed(aircraft_weight, Wing_Total_Area, density, Wing_Lift_Coefficient):

  Flight_Speed_Matrix = []

  for i in range(0, len(Wing_Lift_Coefficient)):

    speed_i = np.sqrt(((aircraft_weight*2)/(density*Wing_Lift_Coefficient[i]*Wing_Total_Area)))
    Flight_Speed_Matrix = np.append(Flight_Speed_Matrix, speed_i)
    print(f'For Wing Lift Coefficient of {Wing_Lift_Coefficient[i]}, the flight speed is {speed_i} [m/s]')

  return Flight_Speed_Matrix

# Create a function to convert dimensionless CL to N

def convert_dimCL_to_newton(density, CL, Wing_Area_S, Glider_Speed):

  Wing_CL_Newton = []

  for i in range(0, len(Glider_Speed)):

    cl_i = (0.5*density*(Glider_Speed[i])**2)*Wing_Area_S*CL[i]
    Wing_CL_Newton = np.append(Wing_CL_Newton, cl_i)

  return Wing_CL_Newton

# Create a function to convert dimensionless CL to N

def convert_dimCD_to_newton(CD, density, Wing_Area_S, Glider_Speed):

  Wing_CD_Newton = []

  for i in range(0, len(Glider_Speed)):

    cd_i = (0.5*density*(Glider_Speed[i])**2)*Wing_Area_S*CD[i]
    Wing_CD_Newton = np.append(Wing_CD_Newton, cd_i)

  return Wing_CD_Newton

# Def a function that returns the total aircraft drag matrix

def get_total_drag(CD_profile_drag_matrix, CD_induced_drag_matrix, CD_fuselage_drag_matrix, CD_profile_drag_matrix_winglet):

    Total_Wing_Drag = np.array(CD_profile_drag_matrix + CD_induced_drag_matrix + CD_fuselage_drag_matrix + CD_profile_drag_matrix_winglet)

    return Total_Wing_Drag

# Create Thermal Model

def get_Thermal_Strenght(Thermal_Type = 'A1',
                         Custom_Vertical_Speed = 3,
                         Custom_Radius = 60,
                         Custom_Gradient = 0.025,
                         Flight_Radius = np.array([50, 60, 70, 80, 90, 100, 150, 400])):

    if Thermal_Type == 'A1':
        Vertical_Speed = 1.75
        rad = 60
        gradient = 0.025
        VT = []
        for fligt_r in Flight_Radius:
            VT_i = (-gradient*fligt_r)+(Vertical_Speed+gradient*rad)

            if VT_i>=0:
                VT = np.append(VT, VT_i)
            else:
                VT = np.append(VT, 0)

    elif Thermal_Type == 'A2':
        Vertical_Speed = 3.5
        rad = 60
        gradient = 0.032
        VT = []
        for fligt_r in Flight_Radius:
            VT_i = (-gradient*fligt_r)+(Vertical_Speed+gradient*rad)

            if VT_i>=0:
                VT = np.append(VT, VT_i)
            else:
                VT = np.append(VT, 0)

    elif Thermal_Type == 'B1':
        Vertical_Speed = 1.75
        rad = 60
        gradient = 0.0045
        VT = []
        for fligt_r in Flight_Radius:
            VT_i = (-gradient*fligt_r)+(Vertical_Speed+gradient*rad)

            if VT_i>=0:
                VT = np.append(VT, VT_i)
            else:
                VT = np.append(VT, 0)

    elif Thermal_Type == 'B2':
        Vertical_Speed = 3.5
        rad = 60
        gradient = 0.006
        VT = []
        for fligt_r in Flight_Radius:
            VT_i = (-gradient*fligt_r)+(Vertical_Speed+gradient*rad)

            if VT_i>=0:
                VT = np.append(VT, VT_i)
            else:
                VT = np.append(VT, 0)

    elif Thermal_Type == 'Custom':
        Vertical_Speed = Custom_Vertical_Speed
        rad = Custom_Radius
        gradient = Custom_Gradient
        VT = []
        for fligt_r in Flight_Radius:
            VT_i = (-gradient*fligt_r)+(Vertical_Speed+gradient*rad)

            if VT_i>=0:
                VT = np.append(VT, VT_i)
            else:
                VT = np.append(VT, 0)

    else:

        VT = 'Non-Computable from given inputs'

        print('Please ensure input is valid i.e A1, B1, A2, B2 or Custom')

    print(f'The vertical speed of thermal is {VT}')

    return VT

# Create Dataframe output function

def create_df(Chord_Lenght, Station_Position, Reynolds_Number, Station_Lift, A_Coefficients ):




    return


# Thermaling Functions

# Calculate Thermaling Glider Velocity

def get_Glider_Thermaling_Velocity(Thermal_Radius = np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                                   Glider_Lift_Coefficient = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4]),
                                   Aircraft_Weight = 5150.25,
                                   Aircraft_Surface_Area = 10.00596,
                                   Density = 1.225
                                   ):

    Velocity = []

    for i in range(0, len(Thermal_Radius)):

        for j in range(0,len(Glider_Lift_Coefficient)):

            if ( 2 * Aircraft_Weight ) / ( Density * Aircraft_Surface_Area * Thermal_Radius[i] * 9.81 * Glider_Lift_Coefficient[j] ) > 1:

                alfa = 89*(np.pi/180)

            else:

                alfa = np.arcsin(( 2 * Aircraft_Weight ) / ( Density * Aircraft_Surface_Area * Thermal_Radius[i] * 9.81 * Glider_Lift_Coefficient[j] ))

            zz = np.sqrt(Thermal_Radius[i] * 9.81 * np.tan(alfa))

            if zz > 55:

                Velocity = np.append(Velocity, 0)
                # Velocity = np.append(Velocity, zz)

            else:

                Velocity = np.append(Velocity, zz)

    #Velocity = np.reshape(Velocity, (len(Thermal_Radius), len(Glider_Lift_Coefficient)))

    print('--- --- ---Thermaling VELOCITY--- --- ---')
    print(Velocity)

    return Velocity

# Interpolate for total aircraft drag

def total_drag_interpolater(glider_speed_matrix, glider_drag_matrix, calculated_speed_matrix):

  total_glider_drag = []

  for i in range(0, len(calculated_speed_matrix)):

    bc = barycentric_interpolate(glider_speed_matrix , glider_drag_matrix, calculated_speed_matrix[i])

    drag_i = bc

    total_glider_drag = np.append(total_glider_drag, drag_i)

    print(f'For Glider Calculated Speed of {calculated_speed_matrix[i]}, the Glider Drag is {drag_i} Newton')

  return total_glider_drag



def total_drag_from_lift_interpolater(glider_lift_matrix, glider_drag_matrix, calculated_lift_matrix):

  total_glider_lift = []

  for i in range(0, len(calculated_lift_matrix)):

    bc = barycentric_interpolate(glider_lift_matrix , glider_drag_matrix, calculated_lift_matrix[i])

    lift_i = bc

    total_glider_lift = np.append(total_glider_lift, lift_i)

  return total_glider_lift



def drag_curve_poly_fit(glider_speed_matrix, glider_drag_matrix, calculated_speed_matrix):

    z = np.polyfit(glider_speed_matrix, glider_drag_matrix, 3)

    p = np.poly1d(z)

    drag_i = p(calculated_speed_matrix)

    drag = []

    for i in range(0,len(calculated_speed_matrix)):

        if calculated_speed_matrix[i] == 0.:
            drag = np.append(drag, 0)

        else:
            drag = np.append(drag, drag_i[i])

    # plt.scatter(calculated_speed_matrix, drag)
    # plt.plot(glider_speed_matrix, glider_drag_matrix)
    # plt.plot(glider_speed_matrix, p(glider_speed_matrix))
    # plt.show()


    return drag

def get_Glider_Thermaling_Induced_Drag(Thermal_Radius = np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                                       Glider_Lift_Coefficient = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4]),
                                       Wing_Efficiency = 0.9,
                                       Aspect_Ratio = 22.6):

    d_induced = []
    for i in range(0,len(Thermal_Radius)):
        for j in range(0,len(Glider_Lift_Coefficient)):

            d_i = Glider_Lift_Coefficient[j]**2 / ( np.pi * Aspect_Ratio * Wing_Efficiency)

            d_induced = np.append(d_induced, d_i)

    return d_induced

def get_Thermaling_Angles(Thermal_Radius = np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                                   Glider_Lift_Coefficient = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4]),
                                   Aircraft_Weight = 5150.25,
                                   Aircraft_Surface_Area = 10.00596,
                                   Density = 1.225):

    Thermaling_Angles = []

    for i in range(0, len(Thermal_Radius)):

        for j in range(0,len(Glider_Lift_Coefficient)):

            if ( 2 * Aircraft_Weight ) / ( Density * Aircraft_Surface_Area * Thermal_Radius[i] * 9.81 * Glider_Lift_Coefficient[j] ) >= 1:

                alfa_ti = 0

            else:

                alfa_ti = np.arcsin(( 2 * Aircraft_Weight ) / ( Density * Aircraft_Surface_Area * Thermal_Radius[i] * 9.81 * Glider_Lift_Coefficient[j] ))

            Thermaling_Angles = np.append(Thermaling_Angles, alfa_ti*180/np.pi )

    print('------------- THERMALING ANGLES -------------')
    print(Thermaling_Angles)


    return Thermaling_Angles

def get_Sink_Speed_Circling(Total_Drag_Matrix,
                            Thermal_Radius = np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                            Glider_Lift_Coefficient = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4]),
                            Aircraft_Weight = 5150.25,
                            Aircraft_Surface_Area = 10.00596,
                            Density = 1.225,
                            ):

    i = 0
    VSC = []
    Total_Drag_Matrix = np.nan_to_num(Total_Drag_Matrix)
    for j in Thermal_Radius:
        for k in Glider_Lift_Coefficient:

            i += 1

            if Total_Drag_Matrix[i-1] == 0:

                VSC_i = -10

            elif Total_Drag_Matrix[i-1] == np.nan:

                VSC_i = -10

            else:

                VSC_i =  ((-1*Total_Drag_Matrix[i-1])*((k)**(-3/2)))*(((2*Aircraft_Weight)/(Density*Aircraft_Surface_Area))**0.5)*((1-((2*Aircraft_Weight*1)/(Density*Aircraft_Surface_Area*9.81*j*k)))**(-3/4))

                # print(f'Final Sink Speed is {VSC_i}')
                # print(f'Total_Drag is {Total_Drag_Matrix[i-1]}')
                # print(f'Radius is {j}')
                # print(f'Lift Coeff is {k}')
            VSC = np.append(VSC, VSC_i)

    print('------------- Sink Speed While Circling -------------')
    print(VSC)

    return VSC

def get_Glider_Climb_Speed(Sink_Speed_Matrix,
                           Thermal_Type='A1',
                           Custom_Vertical_Speed=3,
                           Custom_Radius=60,
                           Custom_Gradient=0.025,
                           Flight_Radius= np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                           Glider_Lift_Coefficient = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4])
                           ):

    i = 0
    climb_speed = []
    Thermal_Speed_VT = get_Thermal_Strenght(Thermal_Type,
                                            Custom_Vertical_Speed,
                                            Custom_Radius,
                                            Custom_Gradient,
                                            np.array([50, 60, 70, 80, 90, 100, 150, 160, 400]))
    for j in Flight_Radius:
        for k in Glider_Lift_Coefficient:

            #print(f'i is {i}')

            bc = barycentric_interpolate(np.array([50, 60, 70, 80, 90, 100, 150, 160, 400]), Thermal_Speed_VT, j)

            VT_i = bc

            #print(f'VT_i is {VT_i}')

            CS_i = Sink_Speed_Matrix[i] + VT_i

            climb_speed = np.append(climb_speed, CS_i)

            i += 1

    print(f'------------- Climb Speed for Thermal Type {Thermal_Type} -------------')
    print(climb_speed)

    max_climb_speed = np.max(climb_speed)

    print(f'------------- Max Climb Speed for Thermal Type {Thermal_Type} -------------')
    print(max_climb_speed)

    return climb_speed , max_climb_speed

def Glider_Climb_Speed_Simulation(Sink_Speed_Matrix,
                                  Flight_Radius=np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                                  Glider_Lift_Coefficient=np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4])):

    T_A1 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='A1', Flight_Radius=Flight_Radius, Glider_Lift_Coefficient=Glider_Lift_Coefficient)
    T_A2 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='A2', Flight_Radius=Flight_Radius, Glider_Lift_Coefficient=Glider_Lift_Coefficient)
    T_B1 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='B1', Flight_Radius=Flight_Radius,Glider_Lift_Coefficient=Glider_Lift_Coefficient)
    T_B2 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='B2', Flight_Radius=Flight_Radius,Glider_Lift_Coefficient=Glider_Lift_Coefficient)
    T_C3 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='Custom', Flight_Radius=Flight_Radius,Glider_Lift_Coefficient=Glider_Lift_Coefficient, Custom_Vertical_Speed=3,Custom_Radius=60, Custom_Gradient=0.025)
    T_C4 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='Custom', Flight_Radius=Flight_Radius,Glider_Lift_Coefficient=Glider_Lift_Coefficient, Custom_Vertical_Speed=4,Custom_Radius=60, Custom_Gradient=0.025)
    T_C5 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='Custom', Flight_Radius=Flight_Radius,Glider_Lift_Coefficient=Glider_Lift_Coefficient, Custom_Vertical_Speed=5,Custom_Radius=60, Custom_Gradient=0.025)
    T_C6 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='Custom', Flight_Radius=Flight_Radius,Glider_Lift_Coefficient=Glider_Lift_Coefficient, Custom_Vertical_Speed=6,Custom_Radius=60, Custom_Gradient=0.025)
    T_C7 = get_Glider_Climb_Speed(Sink_Speed_Matrix, Thermal_Type='Custom', Flight_Radius=Flight_Radius,Glider_Lift_Coefficient=Glider_Lift_Coefficient, Custom_Vertical_Speed=7,Custom_Radius=60, Custom_Gradient=0.025)

    Max_Climb_Speed_Dict = {'Thermal': ['A1', 'A2', 'B1', 'B2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                            'Max_Climb_Speed':[T_A1[1], T_A2[1], T_B1[1], T_B2[1], T_C3[1], T_C4[1], T_C5[1], T_C6[1], T_C7[1]]}

    Max_Thermal_Speed = pd.DataFrame(Max_Climb_Speed_Dict, columns= ['Thermal', 'Max_Climb_Speed'])

    plt.bar(Max_Thermal_Speed['Thermal'], Max_Thermal_Speed['Max_Climb_Speed'])
    plt.xlabel('Thermal Model')
    plt.ylabel('Maximum Thermal Speed')
    plt.show()

    Climb_Speed_Dict = {'Thermal': ['A1', 'A2', 'B1', 'B2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                        'Climb_Speed': [T_A1[0], T_A2[0], T_B1[0], T_B2[0], T_C3[0], T_C4[0], T_C5[0], T_C6[0], T_C7[0]]}

    Climb_Speed = pd.DataFrame(Climb_Speed_Dict, columns= ['Thermal', 'Climb_Speed'])

    return Max_Thermal_Speed, Climb_Speed

def get_sink_speed(CD_Total,
                   density =1.225,
                   aircraft_weight = 5150.25,
                   Wing_Total_Area=50,
                   Glider_Lift_Coefficient=np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4])):

    V_sink = []
    for j in CD_Total:
        for k in Glider_Lift_Coefficient:

            V_sink_i = (((j)/(k**(3/2)))*(((2*aircraft_weight)/(density*Wing_Total_Area))**0.5))

            V_sink = np.append(V_sink, V_sink_i)

    return V_sink

def get_average_cross_country_speed(CD_Total,
                                    Glider_Lift_Matrix,
                                    Max_climb_speed,
                                    density =1.225,
                                    aircraft_weight = 5150.25,
                                    Wing_Total_Area=50,
                                    Cl =np.array([1.1, 0.4, 0.57, 0.31]),
                                    Weather_Model = np.array([0.08, 0.42, 0.08, 0.42]),
                                    Cross_Country_Distance = 300
                                    ):

    Distance = Cross_Country_Distance*Weather_Model
    Max_climb_speed = Max_climb_speed[:-5]
    Max_climb_speed = Max_climb_speed.iloc[:,-1].values
    Speed = (flight_speed(aircraft_weight, Wing_Total_Area, density, Cl))*3.6
    Drag = total_drag_from_lift_interpolater(glider_lift_matrix=Glider_Lift_Matrix, glider_drag_matrix=CD_Total, calculated_lift_matrix=Cl)
    Lift_to_drag = Cl/Drag

    Height_Required = (Distance*1000)/(Lift_to_drag)
    time_for_climb = (Height_Required)/(Max_climb_speed)
    time_for_run = (Distance/Speed)/(60)/(60)
    total_time_for_phase = time_for_climb+time_for_run

    Average_Cross_Country_Speed = (np.sum(Distance))/((np.sum(total_time_for_phase))/(3600))

    print(Average_Cross_Country_Speed)

    return Average_Cross_Country_Speed

# Test run drag modules

LLT_simulation = LLT_simulation()

Wing_CL = LLT_simulation[1]

print('Wing Lift', Wing_CL)

Glider_Speed = flight_speed(aircraft_weight = 5150.25,
                            Wing_Total_Area=50,
                            density = 1.225,
                            Wing_Lift_Coefficient = Wing_CL)

print('Wing Speed', Glider_Speed)

Glider_Drag_Fuselage = fuselage_drag_interpolater(glider_speed_matrix = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75] ,
                           glider_drag_matrix = [11, 14, 18, 22.24, 28.54, 35.8, 44.108, 53.18, 62.96, 74.02, 87, 99.02],
                           calculated_speed_matrix = Glider_Speed)

print('Wing Fuselage Drag', Glider_Drag_Fuselage)

Wing_CL = convert_dimCL_to_newton(density = 1.225,
                                  CL = Wing_CL,
                                  Wing_Area_S = 50,
                                  Glider_Speed = Glider_Speed)

print('Wing Lift', Wing_CL)

##########
##########
##########
##########
##########
##########  NNNNNBBBBB NOTE - we have a different airfoil for every panel so when converting profile drag to newton we need to us the individual panel areas

CD_i = convert_dimCD_to_newton(CD = LLT_simulation[2],
                               density = 1.225,
                               Wing_Area_S = 50,
                               Glider_Speed = Glider_Speed)

print('Wing CDi', CD_i)

# Functions for winglet and profile drag

# --- --- -- ---
Winglet_Constant = 0.000105
CD_profile_Winglet = convert_dimCD_to_newton(CD = np.zeros(len(CD_i)) + Winglet_Constant,
                               density = 1.225,
                               Wing_Area_S = 50,
                               Glider_Speed = Glider_Speed)

print('Wing CD_profile Winglet', CD_profile_Winglet)

CD_profile = convert_dimCD_to_newton(CD = np.zeros(len(CD_i)) + np.arange(0.0005, 0.005, (.005-.0005)/len(CD_i)),
                               density = 1.225,
                               Wing_Area_S = 50,
                               Glider_Speed = Glider_Speed)

print('Wing CD_profile', CD_profile)
# --- --- -- ---

Total_Aircrfaft_Drag = get_total_drag(CD_profile_drag_matrix = CD_profile,
                                      CD_induced_drag_matrix = CD_i,
                                      CD_fuselage_drag_matrix = Glider_Drag_Fuselage,
                                      CD_profile_drag_matrix_winglet = CD_profile_Winglet)


print(f'Wing Total Drag {Total_Aircrfaft_Drag} Newton')

# Test Plots

plt.plot(Glider_Speed, Total_Aircrfaft_Drag, label = 'Total Aircraft Drag')
plt.plot(Glider_Speed, CD_profile_Winglet, label = 'Total Profile Winglet Drag')
plt.plot(Glider_Speed, CD_i, label = 'Total Induced Drag')
plt.plot(Glider_Speed, Glider_Drag_Fuselage, label = 'Total Glider Fuselage Drag')
plt.plot(Glider_Speed, CD_profile, label = 'Profile Drag')
plt.legend()
plt.grid()
plt.xlabel('Velocity [m/s]')
plt.ylabel('Drag Coefficient [Newton]')
plt.show()

# Test run Thermaling modules

get_Thermal_Strenght(Thermal_Type = 'Custom',
                     Custom_Vertical_Speed = 3,
                     Custom_Radius = 60,
                     Custom_Gradient = 0.025,
                     Flight_Radius = np.array([50, 60, 70, 80, 90, 100, 150, 400, 160]))

Velocity = get_Glider_Thermaling_Velocity()

wing_profile_and_fuselage_drag = drag_curve_poly_fit(glider_speed_matrix = Glider_Speed,
                                                     glider_drag_matrix = Total_Aircrfaft_Drag,
                                                     calculated_speed_matrix = Velocity)

get_Thermaling_Angles()

# Convert total wing drag in newton to dimensionless coeff to get sink speed and climb speed

cd_ = (wing_profile_and_fuselage_drag*2)/(1.225*10.00596*Velocity**2)

VSC = get_Sink_Speed_Circling(Total_Drag_Matrix = cd_,
                              Thermal_Radius = np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                              Glider_Lift_Coefficient = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4]),
                              Aircraft_Weight = 5150.25,
                              Aircraft_Surface_Area = 10.00596,
                              Density = 1.225)

get_Glider_Climb_Speed(Sink_Speed_Matrix = VSC,
                           Thermal_Type='A1',
                           Custom_Vertical_Speed=3,
                           Custom_Radius=60,
                           Custom_Gradient=0.025,
                           Flight_Radius=np.array([58, 64, 70, 80, 90, 100, 120, 140, 160, 400]),
                           Glider_Lift_Coefficient = np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4])
                           )

# Get max climb speed for various thermal models

Glider_Climb_Speed_Simulation = Glider_Climb_Speed_Simulation(VSC)
Glider_Max_Climb_Speed = Glider_Climb_Speed_Simulation[0]

# Calculate avg cc speed for a single case

TC1 = get_Thermal_Strenght(Thermal_Type = 'Custom',
                     Custom_Vertical_Speed = 3,
                     Custom_Radius = 60,
                     Custom_Gradient = 0.025,
                     Flight_Radius = np.array([50, 60, 70, 80, 90, 100, 150, 400, 160]))

VS = get_sink_speed(CD_Total = cd_,
                density =1.225,
                aircraft_weight = 5150.25,
                Wing_Total_Area=50,
                Glider_Lift_Coefficient=np.array([0.9, 1, 1.1, 1.2, 1.3, 1.4]))

# Convert total wing drag in newton to dimensionless coeff to get sink speed and climb speed

cd_total = (Total_Aircrfaft_Drag*2)/(1.225*10*Glider_Speed**2)

get_average_cross_country_speed(CD_Total = cd_total,
                                Glider_Lift_Matrix=LLT_simulation[1],
                                Max_climb_speed = Glider_Max_Climb_Speed,
                                density=1.225,
                                aircraft_weight=5150.25,
                                Wing_Total_Area=50,
                                Cl=np.array([1.1, 0.4, 0.57, 0.31]),
                                Weather_Model=np.array([0.08, 0.42, 0.08, 0.42]),
                                Cross_Country_Distance=300
                                )


# Appendix


Chord_Reynolds = get_Reynolds(np.array([0, 2.200, 1.980, 1.800, 0.890, 0.670]), Free_Stream_Velocity = 65, Density = 1.2253, Dynamic_Viscosity = 0.0000173320)

print('Reynolds chord', Chord_Reynolds)