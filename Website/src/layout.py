"""
Main UI file for displaying content on the website.
It creates and generates necessery components for  "home"
page and "about" page accessible from the sidebar
"""

import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import webbrowser
from model import load_model_components, model_predict
from lime_explainer import lime_explanation


def display_logo_components():
    """
    Display logo components for the main home page.
    """
    _, col2, _ = st.columns([3,6,3])
    with col2:
        # Logo + text
        st.image('./assets/logo.png' , use_column_width = True)
        st.markdown('<h5 style="text-align:center;">An attempt to combat misinformation online</h5>',unsafe_allow_html=True)

        # Display loading message if the model and tokenizer are not yet loaded into this session
        if 'model_initialised' not in st.session_state or 'model_initialised' not in st.session_state:
            gif_runner = st.image('./assets/loading.gif')
            text_runner = st.markdown('<p style="text-align:center;">Loading necessary componenets...</p>',unsafe_allow_html=True)

            # Load model components and record this in session state
            load_model_components()
            st.session_state['model_initialised'] = True
            st.session_state['tokenizer_initialised'] = True

            gif_runner.empty()  
            text_runner.empty()


def display_lime_explanation(input_claim: str):
    """
    Display LIME explanation 
    """
    explanation = lime_explanation(input_claim)

    explanation_placeholder = st.empty()
    placeholder = st.empty()
    with explanation_placeholder:
        with st.expander("See output explanation", expanded = False):
            st.write("""
                This fake news detection model outputs weather it predicts the claim to be true or false and gives its probability estimation. 
                Fruthermore it gives insight into how particular words influenced model's decision and lean more towards fake/true sentiment.
                For more details please see "About" page.

                ***Disclaimer: given output is only an estimate. Use trusted sources to validate news.***
            """)
    with placeholder:
        components.html(explanation, height = 400, scrolling = True)


def display_prediction_result(input_claim: str):
    """
    Display prediction results and corresponding graphics
    """
    prediction, prediction_prob = model_predict(input_claim) # Get the prediction and its probability
    if prediction == True:
        emoji = '✅'
    else:
        emoji = '❌'    
    
    st.write("")
    st.markdown(f"""
        <div style="display: flex; justify-content: center; text-align: center; font-size:30-px; width: 100%;">
            <h5>{emoji}  This claim is considered to be {prediction} with the probability of {str(round(prediction_prob* 100, 1))}%  {emoji} </h5>
        </div>
                """, unsafe_allow_html=True)
    st.write("")

            
def sidebar():
    """
    Create a navigation sidebar
    """
    with st.sidebar:
        choose = option_menu("", ["Home", "About"],
            icons=['house', 'info-circle'],
            menu_icon="app-indicator", default_index=0,
            styles={
                "container": {"padding": "5!important", "background": "rgba(0,0,0,0)"},
                "icon": {"color": "green", "font-size": "30px"}, 
                "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px"},
                "nav-link-selected": {"background": "rgba(0,0,0,0)", "color": 'black'},
            }
        )
    if choose == 'Home':
        homePage()
    elif choose == 'About':
        aboutPage()


def homePage():
    """
    Create home page components
    """
    display_logo_components()

    # Claim to predict veracity
    input_claim = st.text_input("Enter the claim that you want to fact-check")

    if st.button("Validate"):
        if len(input_claim) != 0: # Proceed only if there is some text

            # Get prediction and its graphics
            display_prediction_result(input_claim)

            # Display the gif and a loading text
            gif_runner = st.image('./assets/loading.gif')
            text_runner = st.markdown('<p style="text-align:center;">Generating Explanation...</p>',unsafe_allow_html=True)

            # Get LIME explanation
            display_lime_explanation(input_claim)

            # Clear the gif and loading text
            gif_runner.empty()  
            text_runner.empty()

        elif len(input_claim) == 0:
            st.warning("Input a claim you want to fact check above")
    
    st.markdown('<p style="text-align:center;">Fake Take is just an automated classifier, make sure to use trusted sources for fact-checking news.</p>',unsafe_allow_html=True)
            
            

def aboutPage():
    """
    Create About page components
    """
    # About title
    st.write("###")
    st.markdown('<h5 style="font-size: 40px;">About Fake Take</h5>',unsafe_allow_html=True)
    
    # Description
    st.write("""
        Fake Take is an easy to use web tool for inspecting the results of undergraduate project:
        ***A thorough attempt to enhance fake news detection through unbiased dataset, explainability and BERT-based models*** made by ***Jan Marczak***

        It serves as a place to test and experiment with state of the art deep-learning BERT-base machine 
        learning model for natural language tasks called **RoBERTa**, that was pre trained on lots of data to 
        be able to understand english language. The model was then additionally trained on carefully constructed 
        dataset that combines multiple fake news detection datasets into one, in a thoughtful manner.

        To get more insight into model's black-box-like execution, **LIME** AI Explainability method was applied
        on top, to see how each word influenced model's decision. Red color indicates word's tendency towards
        fake prediction and vice-versa with green as can be seen on this example:
    """) 

    # Lime example
    _, col2, _ = st.columns([3,8,3])
    with col2:
        st.image('./assets/lime_example.png')

    st.write("""
        To learn more about it you can visit its github page linked below or download the project report paper. 
        For any questions or suggestions contact jan.marczak00@gmail.com
    """)

    # GitHub and download report buttons
    col1, col2 = st.columns([1]*1+[1.1])
    with col1:
        if st.button('Visit GitHub'):
            webbrowser.open_new_tab('https://github.com/janmarczak')
    with col2:
        with open("./assets/bspr.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()

            st.download_button(label="Download Project Report",
                                data=PDFbyte,
                                file_name="./assets/bspr.pdf",
                                mime='application/octet-stream') 
