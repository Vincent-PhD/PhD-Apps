import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import streamlit as st

# Streamlit Program

st.title('Linear Lifting Line Wing Performance Calculator')

st.write("""
### Wing Performance for Given Inputs:
""")

# Program inputs

st.sidebar.header('Wing Features')
N = st.sidebar.number_input('Number of wing divisions', 50, 400, 50) # number of segments - 1
S = st.sidebar.number_input('Wing Surface Area [m^2]', 5.00, 200.00, 10.004)  # wing Area (m^2)
AR = st.sidebar.number_input('Wing Aspect Ratio', 5.00, 200.00, 22.60)   # wing Aspect Ratio
alpha_twist = st.sidebar.number_input('Wing Twist Angle [deg]', -10.0, 10.0, 0.0) # twist Angle (deg)
i_w	= st.sidebar.number_input('Wing Angle of Attack [deg]', 0.000, 25.00, 2.00)# wing AoA (deg)

st.sidebar.header("""**Define Station-wise Length**""")

Station_Lenght_1 = st.sidebar.number_input('Station 1 Length [m]', 0.00, 0.00, 0.0)
Station_Lenght_2 = st.sidebar.number_input('Station 2 Length [m]', 0.00, 10.00, 2.2)
Station_Lenght_3 = st.sidebar.number_input('Station 3 Length [m]', 0.00, 10.00, 1.98)
Station_Lenght_4 = st.sidebar.number_input('Station 4 Length [m]', 0.00, 10.00, 1.8)
Station_Lenght_5 = st.sidebar.number_input('Station 5 Length [m]', 0.00, 10.00, 0.89)
Station_Lenght_6 = st.sidebar.number_input('Station 5 Length [m]', 0.00, 10.00, 0.670)

st.sidebar.header("""**Define Station-wise Root-Chord**""")

Root_Chords_1 = st.sidebar.number_input('Station 1 Root-Chord', 0.00, 10.00, 0.8)
Root_Chords_2 = st.sidebar.number_input('Station 2 Root-Chord', 0.00, 10.00, .8)
Root_Chords_3 = st.sidebar.number_input('Station 3 Root-Chord', 0.00, 10.00, .7)
Root_Chords_4 = st.sidebar.number_input('Station 4 Root-Chord', 0.00, 10.00, .535)
Root_Chords_5 = st.sidebar.number_input('Station 5 Root-Chord', 0.00, 10.00, .419)
Root_Chords_6 = st.sidebar.number_input('Station 5 Root-Chord', 0.00, 10.00, .21)

st.sidebar.header("""**Define Station-wise Wing Lift Curve Slope**""")


Station_lift_curve_slope_1 = st.sidebar.number_input('Station 1 Lift Curve Slope', 4.00, 8.00, 6.300)
Station_lift_curve_slope_2 = st.sidebar.number_input('Station 2 Lift Curve Slope', 4.00, 8.00, 6.300)
Station_lift_curve_slope_3 = st.sidebar.number_input('Station 3 Lift Curve Slope', 4.00, 8.00, 6.300)
Station_lift_curve_slope_4 = st.sidebar.number_input('Station 4 Lift Curve Slope', 4.00, 8.00, 6.300)
Station_lift_curve_slope_5 = st.sidebar.number_input('Station 5 Lift Curve Slope', 4.00, 8.00, 6.300)


Station_lift_curve_slope = np.array([Station_lift_curve_slope_1,
                                     Station_lift_curve_slope_2,
                                     Station_lift_curve_slope_3,
                                     Station_lift_curve_slope_4,
                                     Station_lift_curve_slope_5
                                     ])

st.sidebar.header("""
Define Station-wise Wing Zero-Lift Angle of Attack
""")

Station_Alpha_2d_1 = st.sidebar.number_input('Station 1 Zero-Lift AoA', -5.0, 5.0, -3.0)
Station_Alpha_2d_2 = st.sidebar.number_input('Station 2 Zero-Lift AoA', -5.0, 5.0, -3.0)
Station_Alpha_2d_3 = st.sidebar.number_input('Station 3 Zero-Lift AoA', -5.0, 5.0, -3.0)
Station_Alpha_2d_4 = st.sidebar.number_input('Station 4 Zero-Lift AoA', -5.0, 5.0, -3.0)
Station_Alpha_2d_5 = st.sidebar.number_input('Station 5 Zero-Lift AoA', -5.0, 5.0, -3.0)


Station_Alpha_2d = np.array([Station_Alpha_2d_1,
                             Station_Alpha_2d_2,
                             Station_Alpha_2d_3,
                             Station_Alpha_2d_4,
                             Station_Alpha_2d_5
                             ])



# Define Station length



Station_Lenght = np.array([ Station_Lenght_1, Station_Lenght_2, Station_Lenght_3, Station_Lenght_4, Station_Lenght_5, Station_Lenght_6])

Root_Chords = np.array([Root_Chords_1, Root_Chords_2, Root_Chords_3, Root_Chords_4, Root_Chords_5,Root_Chords_6])

# Initial Calculations

b = np.sum(Station_Lenght)*2                                              # wing span [m]	                                                    # mean Aerodynamic Chord (m)
C_root = Root_Chords[0]            # Root Chord
Theta_Constant = np.pi/(2*N)                                    # Theta partitioning
Alpha_Constant = np.abs(alpha_twist)/(N-1)                      # Alpha partitioning
Number_of_stations = len(Station_Alpha_2d)                      # Number of wing stations

# Generate Theta Distribution

Theta = []
for i in range(0,N):
    Theta_i = Theta_Constant + (i*Theta_Constant)
    Theta = np.append(Theta, Theta_i)

# Generate Alpha Distribution

Alpha = alpha_twist + i_w
for i in range(0,N):
    if i ==  0:
        Alpha_i = Alpha + (Alpha_Constant)
    else:
        Alpha_i = (alpha_twist + i_w) + (i*Alpha_Constant)
    Alpha = np.append(Alpha, Alpha_i)

# Calculate Station Y-Location
Station_position = []
for i in range(0, len(Station_Lenght)):
    if i <= 0:
        Station_Position_i = Station_Lenght[i]
    if i == 1:
        Station_Position_i = (Station_Lenght[i] +
                              Station_Lenght[i-1])
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
                              Station_Lenght[i - 4] )
    if i == 5:
        Station_Position_i = (Station_Lenght[i] +
                              Station_Lenght[i - 1] +
                              Station_Lenght[i - 2] +
                              Station_Lenght[i - 3] +
                              Station_Lenght[i - 4] +
                              Station_Lenght[i - 5])

    Station_position = np.append(Station_position, Station_Position_i)


print('Station Y-Location',Station_position)

# Generate Linearly Spaced Stations

Station_generated = []
for i in range(0,N):
    Gen_i = ((b)/(2))*np.cos(Theta[i])
    Station_generated = np.append(Station_generated, Gen_i)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

# Find nearest value between generated stations and actual station

Nearest_Value = []
for i in range(0,len(Station_Lenght)):
    NV_i = z = find_nearest(Station_generated, Station_position[i])
    Nearest_Value = np.append(Nearest_Value, NV_i)
print('Nearest Value between gen and actual',Nearest_Value)

# Find index value of Nearest Number Between generated Stations and Actual Stations

Index_Value = []
for i in range(0,len(Station_Lenght)):
    IV_i = np.where(Station_generated == Nearest_Value[i])
    Index_Value = np.append(Index_Value, IV_i)

print('Index value', Index_Value)

# Create length scalar for Root Chord division

Length_Scalar = []
for i in range(0,len(Station_Lenght)):
    if i == 0:
        LS_i = 0
    else:
        LS_i = Index_Value[i] - Index_Value[i-1]

        Length_Scalar = np.abs(np.append(Length_Scalar, LS_i))

print('Lenght Scalar', Length_Scalar)

# Create Root Chord Matrix

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


# Generate Station lift curve slope and zero lift angle of attack information

Station_wise_lift_curve_slope = []
Station_wise_zero_lift_aoa = []
for i in range(0,N):
    if i <= ((N/Number_of_stations)*1):
        Station_wise_zero_lift_aoa_i = Station_Alpha_2d[0]
        Station_wise_lift_curve_slope_i = Station_lift_curve_slope[0]

    elif i <= ((N/Number_of_stations)*2):
        Station_wise_zero_lift_aoa_i = Station_Alpha_2d[1]
        Station_wise_lift_curve_slope_i = Station_lift_curve_slope[1]
    elif i <= ((N/Number_of_stations)*3):
        Station_wise_zero_lift_aoa_i = Station_Alpha_2d[2]
        Station_wise_lift_curve_slope_i = Station_lift_curve_slope[2]
    elif i <= ((N/Number_of_stations)*4):
        Station_wise_zero_lift_aoa_i = Station_Alpha_2d[3]
        Station_wise_lift_curve_slope_i = Station_lift_curve_slope[3]
    else:
        Station_wise_zero_lift_aoa_i = Station_Alpha_2d[4]
        Station_wise_lift_curve_slope_i = Station_lift_curve_slope[4]


    Station_wise_zero_lift_aoa = np.append(Station_wise_zero_lift_aoa , Station_wise_zero_lift_aoa_i)
    Station_wise_lift_curve_slope = np.append(Station_wise_lift_curve_slope, Station_wise_lift_curve_slope_i)
mu = []
for i in range(0,N):
    mu_i = MAC[i] * (Station_wise_lift_curve_slope[i]) / (4 * b)
    mu = np.append(mu, mu_i)

LHS = []
for i in range(0,N):
    LHS_i = mu[i]*((Alpha[i]-Station_wise_zero_lift_aoa[i]))/(180/np.pi)
    LHS = np.append(LHS, LHS_i)

RHS = []
for i in range(0,N):
    for j in range(0, N):
        RHS_i = np.sin((2*(j+1)-1)*Theta[i])*(1+(mu[i]*(2*(j+1)-1))/np.sin(Theta[i]))
        RHS = np.append(RHS, RHS_i)

# Reshape Data

RHS = np.reshape(RHS, (N,N))

# Solve for coefficients

A_coeff = np.linalg.solve(RHS, LHS)


# Calculate Wing Gamma

gamma = []
for i in range (2,N):
    gamma_i = (2 * i - 1) * (((A_coeff[i-1]) / (A_coeff[0]))) ** 2
    gamma = np.append(gamma, gamma_i)
    gamma = np.sum(gamma)

# Calculate wing performance and efficiency

Wing_efficiency = 1/(1+gamma)
Wing_Lift = np.pi*AR*A_coeff[0]
Wing_Induced_Drag =(Wing_Lift**2)/(np.pi*AR*Wing_efficiency)

st.write(f'Total wing stations: {np.round(Number_of_stations,0)}')
st.write(f'Wing efficiency: {np.round(Wing_efficiency*100,2)} %')
st.write(f'Wing dimensionless lift: {np.round(Wing_Lift,5)}')
st.write(f'Wing dimensionless induced drag: {np.round(Wing_Induced_Drag,5)}')

# Calculate span wise lift distribution

Local_Wing_Lift = []
for i in range(0,N):
    for j in range(0, N):
        Local_Wing_Lift_i = A_coeff[j]*np.sin((2 * (j+1) - 1)*Theta[i])
        Local_Wing_Lift = np.append(Local_Wing_Lift, Local_Wing_Lift_i)


# Reshape Local Wing Lift Matrix

Local_Wing_Lift = Local_Wing_Lift.reshape(N,N)

# Create summation matrix for span wise lift distribution calculation

local_Gamma_Sum = []
for i in range(0, N):
    Local_Gamma_Sum_i = 4*(np.sum(Local_Wing_Lift[i]))*((b)/(MAC[i]))
    print(f'Wing local lift for station {np.abs(i-N)} is {Local_Gamma_Sum_i}')

    local_Gamma_Sum = np.append(local_Gamma_Sum, Local_Gamma_Sum_i)

# Show Ideal vs Actual Lift Distribution

Comparitive_Lift_Distribution = st.checkbox('Show Ideal vs Actual Lift Distribution')

if Comparitive_Lift_Distribution:

    # Plot Elliptical Distribution (Ideal)
    fig = plt.figure()
    u = 0                   # x-position of the center
    v = 0                   # y-position of the center
    a = b/2                 # radius on the x-axis i.e half span
    z = local_Gamma_Sum[-1] # radius on the y-axis i.e max lift at station


    t = np.linspace(0, np.pi/2, 100)
    plt.plot( u+a*np.cos(t) , v+z*np.sin(t), label = 'Elliptical Distribution' )
    plt.grid(color='lightgray',linestyle='--')
    plt.xlabel('Wing Span [m]')
    plt.ylabel('Local Lift Coefficient')
    plt.title('Ideal vs Actual Station-Wise Lift Distribution')
    plt.legend()

    # Plot Station wise lift distribution

    plt.plot(Station_generated, local_Gamma_Sum, label = 'Actual Lift Distribution', marker = ',')
    plt.legend()
    st.pyplot(fig)

# Wing Plot

fig = plt.figure()
plt.plot(Station_generated, np.zeros(N)*1, label = 'Wing', marker = ',', c = 'darkblue')
plt.plot(np.zeros(N), np.linspace(0,C_root,num=N)*1, marker = ',', c = 'darkblue')
plt.plot(Station_generated, MAC*1, marker = ',', c = 'darkblue')
plt.plot( (np.zeros(N)+Station_generated[0]),np.linspace(0,MAC[0],num=N)*1,  marker = ',', c = 'darkblue')
plt.plot(Station_generated, (local_Gamma_Sum*Station_generated[0]), label = 'Actual Lift Distribution', marker = ',',c = 'darkred')
plt.plot(Station_generated*-1, np.zeros(N)*-1, marker = ',', c = 'darkblue')
plt.plot(np.zeros(N)*-1, np.linspace(0,C_root,num=N)*1, marker = ',', c = 'darkblue')
plt.plot(Station_generated*-1, MAC*1, marker = ',', c = 'darkblue')
plt.plot( (np.zeros(N)+Station_generated[0])*-1,np.linspace(0,MAC[0],num=N)*1,  marker = ',', c = 'darkblue')
plt.plot(Station_generated*-1, (local_Gamma_Sum*Station_generated[0]), marker = ',',c = 'darkred')
plt.grid(color='lightgray',linestyle='--')
plt.xlabel('Wing Span [m]')
plt.ylabel('Lift Distribution')
plt.title('Wing Lift Distribution')
plt.legend()
st.pyplot(fig)

