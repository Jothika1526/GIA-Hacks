import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import GooglePalm
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
# Load preprocessor and model
preprocessor_path = r'preprocessor.pkl'
model_path = r"model_rf.pkl"
preprocessor = joblib.load(preprocessor_path)
model = joblib.load(model_path)

def predict_sustainability(current_device_details):
    # Map the dictionary keys to the expected column names
    column_mapping = {
        'type': 'DeviceType',
        'usage_hours': 'UsageHoursPerDay',
        'energy_consumption': 'EnergyConsumption',
        'user_preference': 'UserPreferences',
        'malfunction_incidents': 'MalfunctionIncidents',
        'device_age': 'DeviceAgeMonths'
    }
    
    # Re-map the current_device_details dictionary
    features = {column_mapping[key]: value for key, value in current_device_details.items()}
    
    # Define the column names as expected by the preprocessor
    column_names = ['DeviceType', 'UsageHoursPerDay', 'EnergyConsumption', 'UserPreferences', 'MalfunctionIncidents', 'DeviceAgeMonths']
    
    # Create DataFrame with required columns in the specified order
    features_df = pd.DataFrame([features], columns=column_names)
    
    # Apply preprocessor and model prediction
    features_preprocessed = preprocessor.transform(features_df)
    prediction = model.predict(features_preprocessed)
    
    return 'Yes' if prediction[0] == 1 else 'No'


# Load the statistics from the Excel sheet
file_path = "LLM_suggestions\\statistics_by_device_type.xlsx"
stats_df = pd.read_excel(file_path, sheet_name='Statistics')

# Define function to get thresholds from the statistics
def get_thresholds(df, feature):
    percentiles = df[df['Feature'] == feature][['DeviceType', '75th Percentile']].set_index('DeviceType')['75th Percentile'].to_dict()
    return percentiles

# Define thresholds for each feature
energy_thresholds = get_thresholds(stats_df, 'EnergyConsumption')
usage_hours_thresholds = get_thresholds(stats_df, 'UsageHoursPerDay')
malfunction_thresholds = get_thresholds(stats_df, 'MalfunctionIncidents')
device_age_thresholds = get_thresholds(stats_df, 'DeviceAgeMonths')

# Define the rules based on thresholds
def apply_rules(appliance_details):
    device_type = appliance_details['type']
    
    energy_threshold = energy_thresholds.get(device_type, None)
    usage_hours_threshold = usage_hours_thresholds.get(device_type, None)
    malfunction_threshold = malfunction_thresholds.get(device_type, None)
    device_age_threshold = device_age_thresholds.get(device_type, None)
    
    suggestions = []
    
    # Single Feature Rules
    if appliance_details['energy_consumption'] > energy_threshold:
        suggestions.append("Reduce usage or consider replacing with a more energy-efficient model.")
    
    if appliance_details['usage_hours'] > usage_hours_threshold:
        suggestions.append("Reduce usage hours or automate the device to turn off when not in use.")
    
    if appliance_details['malfunction_incidents'] > malfunction_threshold:
        suggestions.append("Regular maintenance or replacing the device may be needed.")
    
    if appliance_details['device_age'] > device_age_threshold:
        suggestions.append("Consider replacing the device as older devices are often less efficient.")
    
    # Two Feature Combinations
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['usage_hours'] > usage_hours_threshold):
        suggestions.append("Significantly reduce usage hours or replace with a more energy-efficient model.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold):
        suggestions.append("Regular maintenance and consider replacing the device to improve efficiency.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['device_age'] > device_age_threshold):
        suggestions.append("Replace the device with a more energy-efficient model.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Optimize settings or use the device more efficiently.")
    
    if (appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold):
        suggestions.append("Reduce usage hours and perform regular maintenance.")
    
    if (appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['device_age'] > device_age_threshold):
        suggestions.append("Reduce usage hours and consider replacing the device.")
    
    if (appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Reduce usage hours and optimize settings.")
    
    if (appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['device_age'] > device_age_threshold):
        suggestions.append("Replace the device due to frequent malfunctions and old age.")
    
    if (appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Regular maintenance and optimize settings.")
    
    if (appliance_details['device_age'] > device_age_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Replace the device and optimize settings.")
    
    # Three Feature Combinations
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold):
        suggestions.append("Reduce usage hours, perform regular maintenance, and consider replacing the device.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['device_age'] > device_age_threshold):
        suggestions.append("Reduce usage hours and replace the device.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Reduce usage hours, optimize settings, and consider replacing the device.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['device_age'] > device_age_threshold):
        suggestions.append("Perform regular maintenance and replace the device.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Perform regular maintenance and optimize settings.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['device_age'] > device_age_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Replace the device and optimize settings.")
    
    if (appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['device_age'] > device_age_threshold):
        suggestions.append("Reduce usage hours, perform regular maintenance, and consider replacing the device.")
    
    if (appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Reduce usage hours, perform regular maintenance, and optimize settings.")
    
    if (appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['device_age'] > device_age_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Reduce usage hours, replace the device, and optimize settings.")
    
    if (appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['device_age'] > device_age_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Perform regular maintenance, replace the device, and optimize settings.")
    
    # Four Feature Combinations
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['device_age'] > device_age_threshold):
        suggestions.append("Reduce usage hours, perform regular maintenance, replace the device, and optimize settings.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Reduce usage hours, perform regular maintenance, and optimize settings.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['device_age'] > device_age_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Reduce usage hours, replace the device, and optimize settings.")
    
    if (appliance_details['energy_consumption'] > energy_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['device_age'] > device_age_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Perform regular maintenance, replace the device, and optimize settings.")
    
    if (appliance_details['usage_hours'] > usage_hours_threshold and 
        appliance_details['malfunction_incidents'] > malfunction_threshold and 
        appliance_details['device_age'] > device_age_threshold and 
        appliance_details['user_preference'] == 1):
        suggestions.append("Reduce usage hours, perform regular maintenance, replace the device, and optimize settings.")
    
    return suggestions

# Google PaLM API key
api_key = "AIzaSyCQCpzFPHgOLhA-YvnPmLSIgzJI3Tnnf0k"
llm = GooglePalm(google_api_key=api_key, temperature=0.7)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["type", "usage_hours", "energy_consumption", "user_preference", "malfunction_incidents", "device_age", "existing_suggestions"],
    template=(
        "The user has a {type} that runs for {usage_hours} hours a day and consumes "
        "{energy_consumption} kWh per day. The user preference for device usage is {user_preference}. "
        "The device has {malfunction_incidents} malfunction incidents and is {device_age} months old. "
        "The appliance is not sustainable. Existing suggestions are: {existing_suggestions}. "
        "Based on this information and your knowledge, please suggest additional ways to make the appliance more sustainable.Consider energy efficiency, usage habits, maintenance practices, and any other relevant factors."
    )
)

# Create the LLM chain
sustainability_chain = LLMChain(llm=llm, prompt=prompt_template)

# Function to get suggestions
def get_sustainability_suggestions(appliance_details):
    # Apply rule-based suggestions
    rule_based_suggestions = apply_rules(appliance_details)
    
    # Format the rule-based suggestions into a single string
    existing_suggestions = "; ".join(rule_based_suggestions)
    
    # Generate AI-based suggestions including the existing suggestions in the prompt
    response = sustainability_chain.run({
        "type": appliance_details['type'],
        "usage_hours": appliance_details['usage_hours'],
        "energy_consumption": appliance_details['energy_consumption'],
        "user_preference": appliance_details['user_preference'],
        "malfunction_incidents": appliance_details['malfunction_incidents'],
        "device_age": appliance_details['device_age'],
        "existing_suggestions": existing_suggestions
    })
    
    return response



# Streamlit application
st.title('Smart Home Appliance Sustainability Checker')

# Initialize session state variables
if 'device_count' not in st.session_state:
    st.session_state.device_count = 1
if 'appliance_details_list' not in st.session_state:
    st.session_state.appliance_details_list = []
if 'buttons_visible' not in st.session_state:
    st.session_state.buttons_visible = True

# Device inputs
if st.session_state.buttons_visible:
    for i in range(st.session_state.device_count):
        st.header(f'Device {i + 1}')
        # device_details = st.session_state.appliance_details_list[i]
        appliance_type = st.selectbox('Device Type', ['Smart Speaker', 'Camera', 'Security System', 'Thermostat', 'Lights'], key=f'type_{i}')
        usage_hours = st.number_input('Usage Hours Per Day', min_value=0, max_value=24, value=1, key=f'usage_{i}')
        energy_consumption = st.number_input('Daily Energy Consumption (kWh)', min_value=0.0, step=0.1, value=0.1, key=f'energy_{i}')
        user_preference = st.selectbox('User Preference for Device Usage', [0, 1], key=f'preference_{i}')
        malfunction_incidents = st.number_input('Number of Malfunction Incidents Reported', min_value=0, value=0, key=f'malfunction_{i}')
        device_age = st.number_input('Device Age in Months', min_value=0, value=0, key=f'age_{i}')
        

    # Add device button
    if st.button('Add Device'):
        appliance_details = {
            'type': st.session_state[f'type_{st.session_state.device_count - 1}'],
            'usage_hours': st.session_state[f'usage_{st.session_state.device_count - 1}'],
            'energy_consumption': st.session_state[f'energy_{st.session_state.device_count - 1}'],
            'user_preference': st.session_state[f'preference_{st.session_state.device_count - 1}'],
            'malfunction_incidents': st.session_state[f'malfunction_{st.session_state.device_count - 1}'],
            'device_age': st.session_state[f'age_{st.session_state.device_count - 1}'],
            
        }
        # Predict sustainability using ML model
        ml_prediction = predict_sustainability(appliance_details)
        is_sustainable = ml_prediction  # Assuming the prediction returns a boolean indicating sustainability status
    
         # Include the sustainability status in the device details
        appliance_details['is_sustainable'] = is_sustainable

        st.session_state.appliance_details_list.append(appliance_details)
        st.session_state.device_count += 1
        st.experimental_rerun()

    # Generate suggestions button
    if st.button('Generate Suggestions'):
        # Add the current device details to the list before generating suggestions
        current_device_details = {
            'type': st.session_state[f'type_{st.session_state.device_count - 1}'],
            'usage_hours': st.session_state[f'usage_{st.session_state.device_count - 1}'],
            'energy_consumption': st.session_state[f'energy_{st.session_state.device_count - 1}'],
            'user_preference': st.session_state[f'preference_{st.session_state.device_count - 1}'],
            'malfunction_incidents': st.session_state[f'malfunction_{st.session_state.device_count - 1}'],
            'device_age': st.session_state[f'age_{st.session_state.device_count - 1}'],
           
        }

        # Predict sustainability using ML model
        ml_prediction = predict_sustainability(current_device_details)
        is_sustainable = ml_prediction  # Assuming the prediction returns a boolean indicating sustainability status
    
         # Include the sustainability status in the device details
        current_device_details['is_sustainable'] = is_sustainable
    
        # Add the updated device details to the list
        st.session_state.appliance_details_list.append(current_device_details)

        # Hide buttons after generating suggestions
        st.session_state.buttons_visible = False
        st.experimental_rerun()
        
# Initialize session state variable for visualization control 
if 'visualization_done' not in st.session_state:
    st.session_state.visualization_done = False
plt.style.use('ggplot')

   
if st.button('Show Usage Time Distribution'):
    st.subheader('Device Usage Time Distribution')
    
    labels = [f'Device {idx+1}' for idx in range(len(st.session_state.appliance_details_list))]
    usage_hours = [device['usage_hours'] for device in st.session_state.appliance_details_list]
    
    fig, ax = plt.subplots(figsize=(2, 2))  # Adjust size as needed
   
    # Create pie chart with colors
    colors = ['#41afaa', '#466eb4', '#00a0e1', '#e6a532', '#d7642c','#af4b91']

    wedges, texts, autotexts = ax.pie(
        usage_hours, 
        labels=labels, 
        autopct='%1.1f%%', 
        colors=colors[:len(labels)], 
        startangle=90,
        textprops={'color': 'white', 'fontsize': 5},
        radius = 0.1 # Adjust font size here
    )
    
    # Optional: Adjust font size of the percentage texts (autotexts)
    for autotext in autotexts:
        autotext.set_fontsize(5)  # Adjust the font size for the percentage texts

    # Set background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Adjust text colors
    for text in texts:
        text.set_color('white')
    for autotext in autotexts:
        autotext.set_color('white')
    
    ax.axis('equal')
    
    st.pyplot(fig)


if st.button('Look at Device Age vs. Malfunction Rate'):
    device_ages = [device['device_age'] for device in st.session_state.appliance_details_list]
    malfunctions = [device['malfunction_incidents'] for device in st.session_state.appliance_details_list]

    colors = ['#41afaa', '#466eb4', '#00a0e1', '#e6a532', '#d7642c','#af4b91']


    fig, ax = plt.subplots()
    for idx, (age, malfunction) in enumerate(zip(device_ages, malfunctions)):
        ax.scatter(age, malfunction, color=colors[idx % len(colors)], edgecolor='black', s=100, label=f'Device {idx + 1}')
    
    ax.set_xlabel('Device Age (months)')
    ax.set_ylabel('Malfunction Incidents')
    ax.set_title('Device Age vs. Malfunction Rate')

    # Set the background color to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Remove grid lines
    ax.grid(False)

    # Add legend to differentiate devices
    ax.legend(facecolor='white', framealpha=1, edgecolor='black')

    st.pyplot(fig)
    
if st.button('Show Energy Consumption Distribution'):
    st.subheader('Energy Consumption Distribution')

    labels = [f'Device {idx+1}' for idx in range(len(st.session_state.appliance_details_list))]
    energy_consumption = [device['energy_consumption'] for device in st.session_state.appliance_details_list]

    # Reduce the size of the pie chart
    fig, ax = plt.subplots(figsize=(2, 2))  # Decrease figsize to make the pie chart smaller

    # Create pie chart with adjusted label colors and font size
    colors = ['#41afaa', '#466eb4', '#00a0e1', '#e6a532', '#d7642c','#af4b91']

    wedges, texts, autotexts = ax.pie(
        energy_consumption,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors[:len(labels)],
        startangle=90,
        textprops={'color': 'white', 'fontsize': 5},  # Adjust font size here
        radius=0.1
    )

    # Optional: Adjust font size of the percentages (autotexts)
    for autotext in autotexts:
        autotext.set_fontsize(5)  # Adjust the font size for the percentage texts

    # Set background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    ax.axis('equal')

    st.pyplot(fig, bbox_inches='tight')

# Prepare data for a stacked bar chart
device_types = [device['type'] for device in st.session_state.appliance_details_list]
user_preferences = [device['user_preference'] for device in st.session_state.appliance_details_list]

# Create a DataFrame for easier manipulation
data = pd.DataFrame({
    'Device Type': device_types,
    'User Preference': user_preferences
})

# Plot the stacked bar chart using seaborn
if st.button('Show User Preference Distribution'):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Device Type', y='User Preference', data=data, errorbar=None, palette="YlOrRd", ax=ax)
    
    # Customize the plot
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_ylabel('User Preference', color='#ffa600')
    ax.set_xlabel('Device Type', color='#ffa600')
    
    ax.tick_params(axis='x', colors='#ffa600')
    ax.tick_params(axis='y', colors='#ffa600')
    
    # Show the plot
    st.pyplot(fig)


# Define the comparative prompt template
comparative_prompt_template = PromptTemplate(
    input_variables=["devices_details"],
    template=(
        "The user has entered details for multiple devices in their smart home. Here are the details for each device:\n"
        "{devices_details}\n"
        "Based on this information and your knowledge, please provide a comparative analysis and comparitive suggestions among the appliances to make them more sustainable and have a sustainable smart home.Dont give individual application sustainability suggestions. Consider energy efficiency, usage habits, maintenance practices, and any other relevant factors."
    )
)

# Create the LLM chain for comparative analysis
comparative_sustainability_chain = LLMChain(llm=llm, prompt=comparative_prompt_template)

# Function to get comparative suggestions
def get_comparative_suggestions(devices_details_list):
    # Format the devices details into a single string
    devices_details_str = "\n".join(
        [f"Device {index + 1} ({device['type']}):\n"
         f"Usage Hours Per Day: {device['usage_hours']}\n"
         f"Daily Energy Consumption (kWh): {device['energy_consumption']}\n"
         f"User Preference for Device Usage: {device['user_preference']}\n"
         f"Number of Malfunction Incidents Reported: {device['malfunction_incidents']}\n"
         f"Device Age in Months: {device['device_age']}\n"
         f"Sustainable: {device['is_sustainable']}"
         for index, device in enumerate(devices_details_list)]
    )
    
    # Generate AI-based comparative suggestions
    response = comparative_sustainability_chain.run({
        "devices_details": devices_details_str
    })
    
    return response

# Define the prompt template for alternative recommendations
alternative_prompt_template = PromptTemplate(
    input_variables=["type", "usage_hours", "energy_consumption", "user_preference", "malfunction_incidents", "device_age"],
    template=(
        "The user has a {type} that runs for {usage_hours} hours a day and consumes "
        "{energy_consumption} kWh per day. The user preference for device usage is {user_preference}. "
        "The device has {malfunction_incidents} malfunction incidents and is {device_age} months old. "
        "This device is not sustainable. Based on this information, please suggest a more sustainable alternative device "
        "that would be more energy-efficient and better suited for the user's needs. Provide details on why this alternative is better."
    )
)

# Create the LLM chain for alternative recommendations
alternative_recommendations_chain = LLMChain(llm=llm, prompt=alternative_prompt_template)

# Function to get alternative recommendations
def get_alternative_recommendations(appliance_details):
    # Generate AI-based alternative recommendations
    response = alternative_recommendations_chain.run({
        "type": appliance_details['type'],
        "usage_hours": appliance_details['usage_hours'],
        "energy_consumption": appliance_details['energy_consumption'],
        "user_preference": appliance_details['user_preference'],
        "malfunction_incidents": appliance_details['malfunction_incidents'],
        "device_age": appliance_details['device_age']
    })
    
    return response

# Generate suggestions output
if not st.session_state.buttons_visible:
    if st.session_state.appliance_details_list:
        for index, appliance_details in enumerate(st.session_state.appliance_details_list):
            st.subheader(f"Device {index + 1}: {appliance_details['type']}")
            if appliance_details['is_sustainable'] == 'Yes':
                st.write('This appliance is sustainable. Keep up the good work!')
            else:
                st.write(appliance_details['is_sustainable'])
                suggestions = get_sustainability_suggestions(appliance_details)
                st.subheader('Suggestions to Make the Appliance More Sustainable:')
                st.write(suggestions)

                # Get and display alternative recommendations
                alternative_recommendations = get_alternative_recommendations(appliance_details)
                st.subheader('Alternative Recommendations:')
                st.write(alternative_recommendations)
    else:
        st.write("No devices added yet.")
    
    if len(st.session_state.appliance_details_list) > 1:
        if st.session_state.appliance_details_list:
            comparative_suggestions = get_comparative_suggestions(st.session_state.appliance_details_list)
            st.subheader('Comparative Analysis and Suggestions:')
            st.write(comparative_suggestions)
        else:
            st.write("Please enter more than one device to compare")
